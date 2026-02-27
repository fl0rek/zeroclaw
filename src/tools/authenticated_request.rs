use super::traits::{Tool, ToolResult};
use crate::config::AuthenticatedRequestConfig;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cached OAuth2 bearer token for a single provider.
struct CachedToken {
    access_token: String,
    /// Adjusted expiry (already accounts for refresh buffer).
    expires_at: Instant,
}

/// HTTP request tool that acquires OIDC client-credentials bearer tokens
/// internally. The agent never sees credentials or tokens — security is
/// enforced by an audience-based URL allowlist instead of SSRF checks.
pub struct AuthenticatedRequestTool {
    security: Arc<SecurityPolicy>,
    config: AuthenticatedRequestConfig,
    /// Per-provider token cache, keyed by provider name.
    token_cache: Arc<RwLock<HashMap<String, CachedToken>>>,
    client: reqwest::Client,
}

impl AuthenticatedRequestTool {
    pub fn new(security: Arc<SecurityPolicy>, config: AuthenticatedRequestConfig) -> Self {
        let timeout = if config.timeout_secs == 0 {
            30
        } else {
            config.timeout_secs
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout))
            .connect_timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .expect("failed to build reqwest client for authenticated_request");
        Self {
            security,
            config,
            token_cache: Arc::new(RwLock::new(HashMap::new())),
            client,
        }
    }

    /// Find the first provider whose `audiences` list contains an origin
    /// matching `request_url`. Returns `(provider_index, matched_audience)`.
    fn find_provider(&self, request_url: &reqwest::Url) -> Option<(usize, String)> {
        let request_origin = origin_of(request_url);
        for (i, provider) in self.config.providers.iter().enumerate() {
            for audience in &provider.audiences {
                if let Ok(aud_url) = reqwest::Url::parse(audience) {
                    if origin_of(&aud_url) == request_origin {
                        return Some((i, audience.clone()));
                    }
                }
            }
        }
        None
    }

    /// Get a valid bearer token for the given provider, using the cache when
    /// possible and refreshing when expired.
    async fn get_token(
        &self,
        provider: &crate::config::OidcProviderConfig,
        audience: &str,
    ) -> anyhow::Result<String> {
        // Fast path: read lock.
        {
            let cache = self.token_cache.read().await;
            if let Some(cached) = cache.get(&provider.name) {
                if Instant::now() < cached.expires_at {
                    return Ok(cached.access_token.clone());
                }
            }
        }

        // Slow path: fetch a new token.
        let token_response = self.fetch_token(provider, audience).await?;

        let access_token = token_response
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("token endpoint did not return access_token"))?
            .to_string();

        let expires_in = token_response
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(3600);

        let buffer = provider.token_refresh_buffer_secs;
        let effective_ttl = expires_in.saturating_sub(buffer);
        let expires_at = Instant::now() + Duration::from_secs(effective_ttl);

        {
            let mut cache = self.token_cache.write().await;
            cache.insert(
                provider.name.clone(),
                CachedToken {
                    access_token: access_token.clone(),
                    expires_at,
                },
            );
        }

        Ok(access_token)
    }

    /// POST to the provider's token endpoint using client credentials.
    async fn fetch_token(
        &self,
        provider: &crate::config::OidcProviderConfig,
        audience: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let scopes = provider.scopes.join(" ");
        let response = self
            .client
            .post(&provider.token_endpoint)
            .basic_auth(&provider.client_id, Some(&provider.client_secret))
            .form(&[
                ("grant_type", "client_credentials"),
                ("scope", &scopes),
                ("audience", audience),
            ])
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("token endpoint request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "token endpoint returned {status}: {}",
                truncate(&body, 500)
            );
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse token response: {e}"))
    }
}

#[async_trait]
impl Tool for AuthenticatedRequestTool {
    fn name(&self) -> &str {
        "authenticated_request"
    }

    fn description(&self) -> &str {
        "Make authenticated HTTP requests to pre-configured services. \
        Uses OIDC client credentials to obtain bearer tokens automatically — \
        you never need to handle credentials or tokens. \
        Only URLs matching configured audience origins are allowed."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to request (must match a configured audience origin)"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)",
                    "default": "GET"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional extra HTTP headers as key-value pairs"
                },
                "body": {
                    "type": "string",
                    "description": "Optional request body (for POST, PUT, PATCH)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Security gates
        if !self.security.can_act() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Action blocked: autonomy is read-only".into()),
            });
        }

        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Action blocked: rate limit exceeded".into()),
            });
        }

        let raw_url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'url' parameter"))?;

        let method_str = args.get("method").and_then(|v| v.as_str()).unwrap_or("GET");
        let headers_val = args.get("headers").cloned().unwrap_or(json!({}));
        let body = args.get("body").and_then(|v| v.as_str());

        // Parse and validate URL
        let parsed_url = match reqwest::Url::parse(raw_url) {
            Ok(u) => u,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Invalid URL: {e}")),
                });
            }
        };

        if parsed_url.scheme() != "http" && parsed_url.scheme() != "https" {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Only http:// and https:// URLs are allowed".into()),
            });
        }

        // Find matching provider
        let (provider_idx, audience) = match self.find_provider(&parsed_url) {
            Some(pair) => pair,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "No configured provider matches URL origin '{}'",
                        origin_of(&parsed_url)
                    )),
                });
            }
        };

        let provider = &self.config.providers[provider_idx];

        // Acquire bearer token
        let token = match self.get_token(provider, &audience).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to acquire token from '{}': {e}", provider.name)),
                });
            }
        };

        // Build HTTP method
        let method = match method_str.to_uppercase().as_str() {
            "GET" => reqwest::Method::GET,
            "POST" => reqwest::Method::POST,
            "PUT" => reqwest::Method::PUT,
            "DELETE" => reqwest::Method::DELETE,
            "PATCH" => reqwest::Method::PATCH,
            "HEAD" => reqwest::Method::HEAD,
            "OPTIONS" => reqwest::Method::OPTIONS,
            other => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Unsupported HTTP method: {other}")),
                });
            }
        };

        // Build request
        let mut request = self
            .client
            .request(method, raw_url)
            .bearer_auth(&token);

        // Add user-provided headers (block Authorization override)
        if let Some(obj) = headers_val.as_object() {
            for (key, value) in obj {
                if key.to_lowercase() == "authorization" {
                    continue; // Silently skip — we manage auth
                }
                if let Some(str_val) = value.as_str() {
                    request = request.header(key.as_str(), str_val);
                }
            }
        }

        if let Some(body_str) = body {
            request = request.body(body_str.to_string());
        }

        // Execute
        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let status_code = status.as_u16();

                let response_text = match response.text().await {
                    Ok(text) => truncate(&text, 500_000),
                    Err(e) => format!("[Failed to read response body: {e}]"),
                };

                let output = format!(
                    "Status: {} {}\n\n{}",
                    status_code,
                    status.canonical_reason().unwrap_or("Unknown"),
                    response_text
                );

                Ok(ToolResult {
                    success: status.is_success(),
                    output,
                    error: if status.is_client_error() || status.is_server_error() {
                        Some(format!("HTTP {status_code}"))
                    } else {
                        None
                    },
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Request failed: {e}")),
            }),
        }
    }
}

/// Extract the origin (scheme + host + port) from a URL, normalizing default ports.
fn origin_of(url: &reqwest::Url) -> String {
    let scheme = url.scheme();
    let host = url.host_str().unwrap_or("");
    match url.port() {
        Some(port) => {
            let is_default = (scheme == "https" && port == 443)
                || (scheme == "http" && port == 80);
            if is_default {
                format!("{scheme}://{host}")
            } else {
                format!("{scheme}://{host}:{port}")
            }
        }
        None => format!("{scheme}://{host}"),
    }
}

/// Truncate a string to `max_len` characters.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max_len).collect();
        t.push_str("\n\n... [Response truncated] ...");
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuthenticatedRequestConfig, OidcProviderConfig};
    use crate::security::{AutonomyLevel, SecurityPolicy};

    fn test_config(audiences: Vec<String>) -> AuthenticatedRequestConfig {
        AuthenticatedRequestConfig {
            enabled: true,
            providers: vec![OidcProviderConfig {
                name: "test-idp".into(),
                token_endpoint: "https://auth.example.com/api/oidc/token".into(),
                client_id: "test-client".into(),
                client_secret: "test-secret".into(),
                audiences,
                scopes: vec!["test.scope".into()],
                token_refresh_buffer_secs: 30,
            }],
            timeout_secs: 10,
        }
    }

    fn test_tool(audiences: Vec<&str>) -> AuthenticatedRequestTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            ..SecurityPolicy::default()
        });
        let config = test_config(audiences.into_iter().map(String::from).collect());
        AuthenticatedRequestTool::new(security, config)
    }

    // ── Origin matching ──────────────────────────────────────────

    #[test]
    fn origin_of_normalizes_default_https_port() {
        let url = reqwest::Url::parse("https://mealie.example.com:443/api").unwrap();
        assert_eq!(origin_of(&url), "https://mealie.example.com");
    }

    #[test]
    fn origin_of_normalizes_default_http_port() {
        let url = reqwest::Url::parse("http://mealie.example.com:80/api").unwrap();
        assert_eq!(origin_of(&url), "http://mealie.example.com");
    }

    #[test]
    fn origin_of_preserves_non_default_port() {
        let url = reqwest::Url::parse("https://mealie.example.com:8443/api").unwrap();
        assert_eq!(origin_of(&url), "https://mealie.example.com:8443");
    }

    #[test]
    fn origin_of_no_port_specified() {
        let url = reqwest::Url::parse("https://mealie.example.com/api/recipes").unwrap();
        assert_eq!(origin_of(&url), "https://mealie.example.com");
    }

    // ── Provider matching ────────────────────────────────────────

    #[test]
    fn find_provider_matches_audience() {
        let tool = test_tool(vec!["https://mealie.example.com"]);
        let url = reqwest::Url::parse("https://mealie.example.com/api/recipes").unwrap();
        let result = tool.find_provider(&url);
        assert!(result.is_some());
        let (idx, aud) = result.unwrap();
        assert_eq!(idx, 0);
        assert_eq!(aud, "https://mealie.example.com");
    }

    #[test]
    fn find_provider_matches_with_port_normalization() {
        let tool = test_tool(vec!["https://mealie.example.com"]);
        let url = reqwest::Url::parse("https://mealie.example.com:443/api/recipes").unwrap();
        assert!(tool.find_provider(&url).is_some());
    }

    #[test]
    fn find_provider_rejects_unknown_host() {
        let tool = test_tool(vec!["https://mealie.example.com"]);
        let url = reqwest::Url::parse("https://unknown.example.com/api").unwrap();
        assert!(tool.find_provider(&url).is_none());
    }

    #[test]
    fn find_provider_rejects_scheme_mismatch() {
        let tool = test_tool(vec!["https://mealie.example.com"]);
        let url = reqwest::Url::parse("http://mealie.example.com/api").unwrap();
        assert!(tool.find_provider(&url).is_none());
    }

    #[test]
    fn find_provider_first_match_wins() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            ..SecurityPolicy::default()
        });
        let config = AuthenticatedRequestConfig {
            enabled: true,
            providers: vec![
                OidcProviderConfig {
                    name: "provider-a".into(),
                    token_endpoint: "https://a.example.com/token".into(),
                    client_id: "a".into(),
                    client_secret: "a".into(),
                    audiences: vec!["https://shared.example.com".into()],
                    scopes: vec![],
                    token_refresh_buffer_secs: 30,
                },
                OidcProviderConfig {
                    name: "provider-b".into(),
                    token_endpoint: "https://b.example.com/token".into(),
                    client_id: "b".into(),
                    client_secret: "b".into(),
                    audiences: vec!["https://shared.example.com".into()],
                    scopes: vec![],
                    token_refresh_buffer_secs: 30,
                },
            ],
            timeout_secs: 10,
        };
        let tool = AuthenticatedRequestTool::new(security, config);
        let url = reqwest::Url::parse("https://shared.example.com/api").unwrap();
        let (idx, _) = tool.find_provider(&url).unwrap();
        assert_eq!(idx, 0, "first provider should win");
    }

    // ── Security gates ───────────────────────────────────────────

    #[tokio::test]
    async fn execute_blocks_readonly_mode() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let config = test_config(vec!["https://example.com".into()]);
        let tool = AuthenticatedRequestTool::new(security, config);
        let result = tool
            .execute(json!({"url": "https://example.com/api"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("read-only"));
    }

    #[tokio::test]
    async fn execute_blocks_when_rate_limited() {
        let security = Arc::new(SecurityPolicy {
            max_actions_per_hour: 0,
            ..SecurityPolicy::default()
        });
        let config = test_config(vec!["https://example.com".into()]);
        let tool = AuthenticatedRequestTool::new(security, config);
        let result = tool
            .execute(json!({"url": "https://example.com/api"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("rate limit"));
    }

    #[tokio::test]
    async fn execute_rejects_unmatched_url() {
        let tool = test_tool(vec!["https://mealie.example.com"]);
        let result = tool
            .execute(json!({"url": "https://unknown.example.com/api"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("No configured provider"));
    }

    #[tokio::test]
    async fn execute_rejects_invalid_url() {
        let tool = test_tool(vec!["https://example.com"]);
        let result = tool
            .execute(json!({"url": "not-a-url"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Invalid URL"));
    }

    #[tokio::test]
    async fn execute_rejects_ftp_scheme() {
        let tool = test_tool(vec!["ftp://example.com"]);
        let result = tool
            .execute(json!({"url": "ftp://example.com/file"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Only http"));
    }

    // ── Truncation ───────────────────────────────────────────────

    #[test]
    fn truncate_within_limit() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn truncate_over_limit() {
        let result = truncate("hello world", 5);
        assert!(result.starts_with("hello"));
        assert!(result.contains("[Response truncated]"));
    }

    // ── Token caching ────────────────────────────────────────────

    #[tokio::test]
    async fn cached_token_is_reused_when_valid() {
        let tool = test_tool(vec!["https://example.com"]);
        // Manually insert a cached token
        {
            let mut cache = tool.token_cache.write().await;
            cache.insert(
                "test-idp".into(),
                CachedToken {
                    access_token: "cached-token".into(),
                    expires_at: Instant::now() + Duration::from_secs(3600),
                },
            );
        }

        let provider = &tool.config.providers[0];
        let token = tool.get_token(provider, "https://example.com").await.unwrap();
        assert_eq!(token, "cached-token");
    }
}
