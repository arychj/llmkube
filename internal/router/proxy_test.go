/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package router

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakeBackend wraps an httptest.Server and a handler that the test can
// swap per case. It also tracks call counts so tests can assert which
// backend(s) were hit.
type fakeBackend struct {
	srv    *httptest.Server
	calls  atomic.Int64
	status atomic.Int64 // status code to return; defaults to 200
	body   atomic.Pointer[string]
	stream atomic.Bool
}

func newFakeBackend(t *testing.T) *fakeBackend {
	t.Helper()
	fb := &fakeBackend{}
	fb.status.Store(200)
	defaultBody := `{"choices":[{"message":{"content":"hi"}}]}`
	fb.body.Store(&defaultBody)

	fb.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fb.calls.Add(1)
		status := int(fb.status.Load())
		body := *fb.body.Load()

		if fb.stream.Load() {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(status)
			flusher, _ := w.(http.Flusher)
			for _, chunk := range []string{"a", "b", "c"} {
				_, _ = fmt.Fprintf(w, "data: {\"delta\":%q}\n\n", chunk)
				if flusher != nil {
					flusher.Flush()
				}
				time.Sleep(2 * time.Millisecond)
			}
			_, _ = fmt.Fprintln(w, "data: [DONE]")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(fb.srv.Close)
	return fb
}

func (fb *fakeBackend) URL() string { return fb.srv.URL }

// proxyHarness sets up a Proxy wired to two fake backends (one local,
// one cloud) plus the standard pii rule. Returns the proxy mounted on a
// fresh ServeMux ready to serve.
type proxyHarness struct {
	cfg       *Config
	localBack *fakeBackend
	cloudBack *fakeBackend
	proxy     *Proxy
	handler   http.Handler
}

func newProxyHarness(t *testing.T) *proxyHarness {
	t.Helper()
	local := newFakeBackend(t)
	cloud := newFakeBackend(t)

	cfg := &Config{
		Backends: []Backend{
			{Name: "local-qwen", Tier: "local", Address: local.URL()},
			{Name: "cloud-opus", Tier: "cloud", Address: cloud.URL(),
				Provider: "anthropic", Model: "claude-opus-4-7"},
		},
		Rules: []Rule{
			{
				Name:       "pii-stays-local",
				Match:      RuleMatch{DataClassification: []string{"pii"}},
				Route:      RuleRoute{Backends: []string{"local-qwen"}},
				FailClosed: true,
			},
			{
				Name:  "complex-to-cloud",
				Match: RuleMatch{TaskComplexity: "complex"},
				Route: RuleRoute{Backends: []string{"cloud-opus", "local-qwen"}},
			},
		},
		DefaultRoute: "local-qwen",
		Policy: Policy{
			Classification: ClassificationPolicy{Mode: "header-only"},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("harness config: %v", err)
	}

	proxy := NewProxy(cfg, slog.Default())
	mux := http.NewServeMux()
	proxy.Mount(mux)
	return &proxyHarness{
		cfg:       cfg,
		localBack: local,
		cloudBack: cloud,
		proxy:     proxy,
		handler:   mux,
	}
}

func (h *proxyHarness) post(t *testing.T, payload map[string]any, headers map[string]string) *http.Response {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, req)
	return rec.Result()
}

// streamingPost uses a real httptest.Server so we exercise streaming via
// a chunked response (httptest.ResponseRecorder doesn't implement
// Flusher).
func (h *proxyHarness) streamingPost(t *testing.T, payload map[string]any, headers map[string]string) *http.Response {
	t.Helper()
	srv := httptest.NewServer(h.handler)
	t.Cleanup(srv.Close)
	body, _ := json.Marshal(payload)
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	return resp
}

func TestProxyHealth(t *testing.T) {
	h := newProxyHarness(t)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("/health = %d, want 200", rec.Code)
	}
}

func TestProxyModels(t *testing.T) {
	h := newProxyHarness(t)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/v1/models = %d, want 200", rec.Code)
	}
	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode /v1/models: %v", err)
	}
	data, _ := got["data"].([]any)
	if len(data) != 2 {
		t.Errorf("expected 2 models, got %d", len(data))
	}
}

// TestProxyModelsPublishesAliases verifies that alias names surface in
// /v1/models alongside backends, and that an alias whose name collides with
// a backend id de-duplicates to a single entry.
func TestProxyModelsPublishesAliases(t *testing.T) {
	local := newFakeBackend(t)
	cfg := &Config{
		Backends: []Backend{
			{Name: "gemma-4-e4b", Tier: "local", Address: local.URL()},
		},
		Aliases: []Alias{
			{Name: "fast", Backends: []string{"gemma-4-e4b"}},
			{Name: "think", Backends: []string{"gemma-4-e4b"}},
			// Collides with the backend id; must not double-publish.
			{Name: "gemma-4-e4b", Backends: []string{"gemma-4-e4b"}},
		},
		DefaultRoute: "gemma-4-e4b",
		Policy:       Policy{Classification: ClassificationPolicy{Mode: "header-only"}},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("config: %v", err)
	}
	mux := http.NewServeMux()
	NewProxy(cfg, slog.Default()).Mount(mux)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/v1/models = %d, want 200", rec.Code)
	}
	var got struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	ids := make(map[string]int)
	for _, m := range got.Data {
		ids[m.ID]++
	}
	for _, want := range []string{"gemma-4-e4b", "fast", "think"} {
		if ids[want] != 1 {
			t.Errorf("model %q count = %d, want 1; full list: %v", want, ids[want], ids)
		}
	}
	// Exactly three distinct ids: the backend plus two non-colliding
	// aliases. Pins the dedup (the gemma-4-e4b alias collapses) and guards
	// against any phantom entry leaking in.
	if len(got.Data) != 3 {
		t.Errorf("expected 3 models, got %d: %v", len(got.Data), ids)
	}
}

// aliasDispatchPost builds a proxy from cfg, POSTs the given model, and
// returns the recorder. Non-streaming, so httptest.NewRecorder suffices.
func aliasDispatchPost(t *testing.T, cfg *Config, model string) *httptest.ResponseRecorder {
	t.Helper()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("config: %v", err)
	}
	mux := http.NewServeMux()
	NewProxy(cfg, slog.Default()).Mount(mux)
	body, _ := json.Marshal(map[string]any{"model": model})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	return rec
}

// TestProxyDispatchesViaAlias is the end-to-end check that a request naming
// an alias actually dispatches through the synthetic alias rule to its
// backend — the one place the alias-as-rule abstraction could leak. The
// default route points at a DIFFERENT backend so that if alias matching
// broke, the request would fall through to the default and the assertions
// would catch it (rather than silently hitting the same backend).
func TestProxyDispatchesViaAlias(t *testing.T) {
	aliasBackend := newFakeBackend(t)
	defaultBackend := newFakeBackend(t)
	cfg := &Config{
		Backends: []Backend{
			{Name: "b-alias", Tier: "local", Address: aliasBackend.URL()},
			{Name: "b-default", Tier: "local", Address: defaultBackend.URL()},
		},
		Aliases:      []Alias{{Name: "fast", Backends: []string{"b-alias"}}},
		DefaultRoute: "b-default",
		Policy:       Policy{Classification: ClassificationPolicy{Mode: "header-only"}},
	}
	rec := aliasDispatchPost(t, cfg, "fast")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if aliasBackend.calls.Load() != 1 {
		t.Errorf("alias backend calls = %d, want 1", aliasBackend.calls.Load())
	}
	if defaultBackend.calls.Load() != 0 {
		t.Errorf("default backend must not be hit when the alias matches; calls = %d", defaultBackend.calls.Load())
	}
}

// TestProxyAliasOrderedFallback exercises an alias's ordered backend list:
// the primary 5xxs and the request must fall over to the secondary. This is
// the alias's reason to exist (local-first, cloud-fallback degradation).
func TestProxyAliasOrderedFallback(t *testing.T) {
	primary := newFakeBackend(t)
	secondary := newFakeBackend(t)
	primary.status.Store(500)
	cfg := &Config{
		Backends: []Backend{
			{Name: "b1", Tier: "local", Address: primary.URL()},
			{Name: "b2", Tier: "local", Address: secondary.URL()},
		},
		Aliases:      []Alias{{Name: "frontier", Backends: []string{"b1", "b2"}}},
		DefaultRoute: "b1",
		Policy:       Policy{Classification: ClassificationPolicy{Mode: "header-only"}},
	}
	rec := aliasDispatchPost(t, cfg, "frontier")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (fallback to secondary)", rec.Code)
	}
	if primary.calls.Load() != 1 {
		t.Errorf("primary should be tried once, calls = %d", primary.calls.Load())
	}
	if secondary.calls.Load() != 1 {
		t.Errorf("secondary should serve as fallback, calls = %d", secondary.calls.Load())
	}
}

func TestProxyRoutesPIIToLocal(t *testing.T) {
	h := newProxyHarness(t)
	resp := h.post(t, map[string]any{"model": "any"}, map[string]string{
		"x-llmkube-classification": "pii",
	})
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if h.localBack.calls.Load() != 1 {
		t.Errorf("local backend calls = %d, want 1", h.localBack.calls.Load())
	}
	if h.cloudBack.calls.Load() != 0 {
		t.Errorf("cloud backend should not be called for pii, calls = %d", h.cloudBack.calls.Load())
	}
}

func TestProxyRoutesComplexToCloud(t *testing.T) {
	h := newProxyHarness(t)
	resp := h.post(t, map[string]any{"model": "any"}, map[string]string{
		"x-llmkube-task-complexity": "complex",
	})
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if h.cloudBack.calls.Load() != 1 {
		t.Errorf("cloud backend calls = %d, want 1", h.cloudBack.calls.Load())
	}
}

func TestProxyFallsThroughToDefault(t *testing.T) {
	h := newProxyHarness(t)
	resp := h.post(t, map[string]any{"model": "any"}, nil)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if h.localBack.calls.Load() != 1 {
		t.Errorf("default should hit local, calls = %d", h.localBack.calls.Load())
	}
}

// TestProxyFailClosedOnLocalDown is the regulated-data gate verified at
// runtime: when the only local backend is unhealthy, a PII request is
// refused with 503 (fail-closed runtime) and the cloud backend is
// never touched.
func TestProxyFailClosedOnLocalDown(t *testing.T) {
	h := newProxyHarness(t)
	h.proxy.disp.MarkUnhealthy("local-qwen")

	resp := h.post(t, map[string]any{"model": "any"}, map[string]string{
		"x-llmkube-classification": "pii",
	})
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 fail-closed runtime", resp.StatusCode)
	}
	if h.cloudBack.calls.Load() != 0 {
		t.Errorf("cloud must not receive pii request; calls = %d", h.cloudBack.calls.Load())
	}
}

// TestProxyNonFailClosedReturns502OnAllBackendsDown documents the
// negative case: a non-fail-closed rule with all backends unreachable
// returns 502 (BadGateway), not 503. This is what distinguishes a
// generic upstream outage from a policy-level refusal.
func TestProxyNonFailClosedReturns502OnAllBackendsDown(t *testing.T) {
	h := newProxyHarness(t)
	// Disable both backends so the default-route fallback fails too.
	h.proxy.disp.MarkUnhealthy("local-qwen")
	h.proxy.disp.MarkUnhealthy("cloud-opus")

	// No classification, no complexity: hits the default route, which
	// the harness configures with FailClosed=false.
	resp := h.post(t, map[string]any{"model": "any"}, nil)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (non-fail-closed upstream outage)", resp.StatusCode)
	}
}

func TestProxyPrimaryFallbackOnUpstream5xx(t *testing.T) {
	h := newProxyHarness(t)
	// Cloud (primary for complex rule) returns 500; proxy must fall
	// over to local (the secondary in the route).
	h.cloudBack.status.Store(500)

	resp := h.post(t, map[string]any{"model": "any"}, map[string]string{
		"x-llmkube-task-complexity": "complex",
	})
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (fallback to local)", resp.StatusCode)
	}
	if h.cloudBack.calls.Load() != 1 {
		t.Errorf("cloud should be tried once, calls = %d", h.cloudBack.calls.Load())
	}
	if h.localBack.calls.Load() != 1 {
		t.Errorf("local should be tried as fallback, calls = %d", h.localBack.calls.Load())
	}
}

func TestProxyStreamingSSE(t *testing.T) {
	h := newProxyHarness(t)
	h.localBack.stream.Store(true)

	resp := h.streamingPost(t, map[string]any{
		"model":  "any",
		"stream": true,
	}, nil)
	defer func() { _ = resp.Body.Close() }()

	if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", got)
	}

	scanner := bufio.NewScanner(resp.Body)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	joined := strings.Join(lines, "\n")
	for _, want := range []string{"delta", "[DONE]"} {
		if !strings.Contains(joined, want) {
			t.Errorf("stream output missing %q; got:\n%s", want, joined)
		}
	}
}

// TestProxy503WhenNoRoute confirms the explicit "no rule and no default"
// case returns 503 rather than panicking.
func TestProxy503WhenNoRoute(t *testing.T) {
	cfg := &Config{
		Backends: []Backend{{Name: "x", Tier: "local", Address: "http://nowhere.invalid"}},
		// No rules, no default route.
	}
	proxy := NewProxy(cfg, slog.Default())
	mux := http.NewServeMux()
	proxy.Mount(mux)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"any"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", rec.Code)
	}
}

// TestProxyRejectsOversizedBody enforces the body-size cap; a request
// larger than maxRequestBodyBytes is rejected with 400.
func TestProxyRejectsOversizedBody(t *testing.T) {
	h := newProxyHarness(t)
	big := strings.Repeat("x", maxRequestBodyBytes+1)
	body := `{"model":"any","padding":"` + big + `"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

// TestExtractFeaturesHandlesMalformedBody ensures a non-JSON body does
// not crash feature extraction; the matcher just sees an empty model.
func TestExtractFeaturesHandlesMalformedBody(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader("not json"))
	f, _ := extractFeatures([]byte("not json"), r, "x-llmkube-classification")
	if f.Model != "" {
		t.Errorf("Model = %q, want empty", f.Model)
	}
}

// TestExtractFeaturesParsesURL is a sanity-check that the helper still
// compiles after refactoring; importing net/url keeps build tags honest.
func TestExtractFeaturesParsesURL(t *testing.T) {
	u, _ := url.Parse("http://x/v1/chat/completions")
	if u.Path != "/v1/chat/completions" {
		t.Errorf("url parse busted: %q", u.Path)
	}
}

// TestResolveDispatchTimeoutPrecedence pins the per-attempt timeout
// resolution order added for #458: rule.Timeout (strictest) wins over
// backend.Timeout wins over the proxy default. Zero values fall
// through so unset fields don't silently shrink the budget.
func TestResolveDispatchTimeoutPrecedence(t *testing.T) {
	cases := []struct {
		name      string
		ruleTO    time.Duration
		backendTO time.Duration
		proxyDef  time.Duration
		want      time.Duration
	}{
		{"rule wins", 5 * time.Second, 30 * time.Second, 120 * time.Second, 5 * time.Second},
		{"backend wins when rule unset", 0, 30 * time.Second, 120 * time.Second, 30 * time.Second},
		{"proxy default when both unset", 0, 0, 120 * time.Second, 120 * time.Second},
		{"rule wins even when shorter than backend", 1 * time.Second, 60 * time.Second, 120 * time.Second, 1 * time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dec := &MatchResult{Rule: &Rule{Name: "r", Timeout: tc.ruleTO}}
			backend := &Backend{Name: "b", Timeout: tc.backendTO}
			got := resolveDispatchTimeout(dec, backend, tc.proxyDef)
			if got != tc.want {
				t.Errorf("resolveDispatchTimeout = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestResolveDispatchTimeoutHandlesNilDecisionOrBackend confirms the
// helper is safe against the corner cases the proxy's dispatch loop
// can plausibly hit: nil MatchResult (no rule matched) or nil chosen
// backend (defensive — shouldn't happen but worth pinning).
func TestResolveDispatchTimeoutHandlesNilDecisionOrBackend(t *testing.T) {
	want := 100 * time.Second
	if got := resolveDispatchTimeout(nil, &Backend{}, want); got != want {
		t.Errorf("nil decision: got %v, want %v", got, want)
	}
	if got := resolveDispatchTimeout(&MatchResult{}, nil, want); got != want {
		t.Errorf("nil backend: got %v, want %v", got, want)
	}
	if got := resolveDispatchTimeout(nil, nil, want); got != want {
		t.Errorf("both nil: got %v, want %v", got, want)
	}
}

// TestResolveDispatchTimeoutAppliesAliasTimeout proves the alias timeout
// reaches dispatch: it flows alias.Timeout -> synthetic rule -> the decision
// resolveDispatchTimeout reads. Driven through the real matcher so it covers
// the whole wiring, not a hand-built decision. Without this, the alias-as-rule
// abstraction is proven only at match time, not at dispatch time.
func TestResolveDispatchTimeoutAppliesAliasTimeout(t *testing.T) {
	cfg := validConfig()
	cfg.Aliases = []Alias{{Name: "fast", Backends: []string{"local-qwen"}, Timeout: 7 * time.Second}}
	m := NewMatcher(cfg)
	dec := m.Match(&RequestFeatures{Model: "fast"})
	// Backend carries no timeout and the proxy default is large, so the alias
	// timeout is the only thing that can produce 7s.
	if got := resolveDispatchTimeout(&dec, m.BackendByName("local-qwen"), 99*time.Second); got != 7*time.Second {
		t.Errorf("alias timeout not applied at dispatch: got %v, want 7s", got)
	}
}

// TestProxyAppliesRuleTimeoutOnDispatch is the end-to-end version of
// the resolve test: a rule with a sub-100ms timeout against a fake
// backend that sleeps 200ms should surface as a context-deadline
// error, with the proxy returning 502 (non-fail-closed) or 503
// (fail-closed). We assert the upstream's call count remains 1
// (the backend WAS reached but the proxy abandoned waiting).
func TestProxyAppliesRuleTimeoutOnDispatch(t *testing.T) {
	slow := newFakeBackend(t)
	slow.srv.Close()
	// Replace the server with one that sleeps before responding so we
	// trigger the per-attempt deadline.
	slow.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		slow.calls.Add(1)
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(slow.srv.Close)

	cfg := &Config{
		Backends: []Backend{{Name: "slow", Tier: "local", Address: slow.srv.URL}},
		Rules: []Rule{{
			Name:    "tight",
			Match:   RuleMatch{Headers: map[string]string{"x-llmkube-task": "code"}},
			Route:   RuleRoute{Backends: []string{"slow"}},
			Timeout: 30 * time.Millisecond,
		}},
		DefaultRoute: "slow",
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("cfg: %v", err)
	}
	proxy := NewProxy(cfg, slog.Default())
	mux := http.NewServeMux()
	proxy.Mount(mux)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"any"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-llmkube-task", "code")
	rec := httptest.NewRecorder()
	start := time.Now()
	mux.ServeHTTP(rec, req)
	elapsed := time.Since(start)

	if rec.Code != http.StatusBadGateway && rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 502 or 503 (deadline-exceeded)", rec.Code)
	}
	// The proxy should give up well before the upstream's 200ms sleep
	// completes. 100ms is generous headroom for test runner overhead.
	if elapsed > 100*time.Millisecond {
		t.Errorf("proxy waited %v on a 30ms rule timeout; deadline not applied", elapsed)
	}
	// The backend WAS contacted but the proxy abandoned.
	if slow.calls.Load() != 1 {
		t.Errorf("backend call count = %d, want 1 (dispatch should reach upstream then time out)",
			slow.calls.Load())
	}
}
