/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package router

import (
	"testing"
	"time"
)

func matcherFromValid() *Matcher {
	return NewMatcher(validConfig())
}

func TestMatchPIIRouteWins(t *testing.T) {
	got := matcherFromValid().Match(&RequestFeatures{Classification: "pii"})
	if got.Rule == nil {
		t.Fatal("expected pii rule to match, got default")
	}
	if got.Rule.Name != "pii-stays-local" {
		t.Errorf("matched rule = %q, want pii-stays-local", got.Rule.Name)
	}
	if !got.FailClosed {
		t.Error("matched fail-closed rule should report FailClosed=true")
	}
	if got.Strategy != strategyPrimaryFallback {
		t.Errorf("default strategy = %q, want primary-fallback", got.Strategy)
	}
}

func TestMatchFallsThroughToDefault(t *testing.T) {
	got := matcherFromValid().Match(&RequestFeatures{Classification: "public"})
	if got.Rule != nil {
		t.Errorf("expected no rule match, got %q", got.Rule.Name)
	}
	if len(got.Backends) != 1 || got.Backends[0] != "local-qwen" {
		t.Errorf("expected default-route fallback to local-qwen, got %v", got.Backends)
	}
}

func TestMatchReturnsEmptyWhenNoDefaultAndNoRule(t *testing.T) {
	cfg := validConfig()
	cfg.Rules = nil
	cfg.DefaultRoute = ""
	got := NewMatcher(cfg).Match(&RequestFeatures{Classification: "public"})
	if len(got.Backends) != 0 {
		t.Errorf("expected no backends, got %v", got.Backends)
	}
}

func TestMatchModelGlob(t *testing.T) {
	cfg := validConfig()
	cfg.Rules = []Rule{{
		Name:  "qwen-family",
		Match: RuleMatch{Models: []string{"qwen3-*"}},
		Route: RuleRoute{Backends: []string{"local-qwen"}},
	}}
	m := NewMatcher(cfg)
	if got := m.Match(&RequestFeatures{Model: "qwen3-coder-30b"}); got.Rule == nil {
		t.Error("qwen3-coder-30b should match qwen3-*")
	}
	if got := m.Match(&RequestFeatures{Model: "llama-3"}); got.Rule != nil {
		t.Errorf("llama-3 should not match qwen3-*, matched %q", got.Rule.Name)
	}
}

func TestMatchHeadersCaseInsensitive(t *testing.T) {
	cfg := validConfig()
	cfg.Rules = []Rule{{
		Name:  "team-rule",
		Match: RuleMatch{Headers: map[string]string{"X-Team": "research"}},
		Route: RuleRoute{Backends: []string{"local-qwen"}},
	}}
	m := NewMatcher(cfg)
	got := m.Match(&RequestFeatures{Headers: map[string]string{"x-team": "research"}})
	if got.Rule == nil {
		t.Error("lowercase header should match canonical declaration")
	}
}

func TestMatchTaskComplexity(t *testing.T) {
	cfg := validConfig()
	cfg.Rules = []Rule{{
		Name:  "complex-to-cloud",
		Match: RuleMatch{TaskComplexity: "complex"},
		Route: RuleRoute{Backends: []string{"cloud-opus"}},
	}}
	m := NewMatcher(cfg)
	if got := m.Match(&RequestFeatures{TaskComplexity: "complex"}); got.Rule == nil {
		t.Error("complex task should match")
	}
	if got := m.Match(&RequestFeatures{TaskComplexity: "simple"}); got.Rule != nil {
		t.Errorf("simple task should not match complex rule, matched %q", got.Rule.Name)
	}
}

func TestMatchRequiredCapabilities(t *testing.T) {
	cfg := validConfig()
	cfg.Backends[0].Capabilities = []string{"code"}
	cfg.Backends[1].Capabilities = []string{"vision", "long-context"}
	cfg.Rules = []Rule{{
		Name: "vision-rule",
		Match: RuleMatch{
			Models:               []string{"*"},
			RequiredCapabilities: []string{"vision"},
		},
		Route: RuleRoute{Backends: []string{"local-qwen", "cloud-opus"}},
	}}
	m := NewMatcher(cfg)
	// At least one route backend has vision; should match.
	if got := m.Match(&RequestFeatures{Model: "any"}); got.Rule == nil {
		t.Error("expected match: cloud-opus has vision")
	}

	// Remove vision from cloud-opus; now no backend in the route has it.
	cfg.Backends[1].Capabilities = []string{"long-context"}
	m = NewMatcher(cfg)
	if got := m.Match(&RequestFeatures{Model: "any"}); got.Rule != nil {
		t.Errorf("expected no match when no route backend has vision; matched %q", got.Rule.Name)
	}
}

func TestMatchFirstRuleWins(t *testing.T) {
	cfg := validConfig()
	cfg.Rules = []Rule{
		{
			Name:  "first",
			Match: RuleMatch{Models: []string{"*"}},
			Route: RuleRoute{Backends: []string{"local-qwen"}},
		},
		{
			Name:  "second",
			Match: RuleMatch{Models: []string{"*"}},
			Route: RuleRoute{Backends: []string{"cloud-opus"}},
		},
	}
	got := NewMatcher(cfg).Match(&RequestFeatures{Model: "any-model"})
	if got.Rule == nil || got.Rule.Name != "first" {
		t.Errorf("expected first rule to win, got %v", got.Rule)
	}
}

func TestMatchAliasByExactName(t *testing.T) {
	cfg := validConfig()
	cfg.Aliases = []Alias{
		{Name: "fast", Backends: []string{"cloud-opus", "local-qwen"}, Timeout: 9 * time.Second},
	}
	m := NewMatcher(cfg)

	got := m.Match(&RequestFeatures{Model: "fast"})
	if got.Rule == nil || got.Rule.Name != "fast" {
		t.Fatalf("expected alias fast to match, got %v", got.Rule)
	}
	if !got.IsAlias {
		t.Error("alias match should set IsAlias=true")
	}
	if got.Strategy != strategyPrimaryFallback {
		t.Errorf("alias strategy = %q, want primary-fallback", got.Strategy)
	}
	if len(got.Backends) != 2 || got.Backends[0] != "cloud-opus" {
		t.Errorf("alias backends = %v, want ordered [cloud-opus local-qwen]", got.Backends)
	}
	if got.Rule.Timeout != 9*time.Second {
		t.Errorf("alias timeout = %s, want 9s", got.Rule.Timeout)
	}

	if got := m.Match(&RequestFeatures{Model: "unknown"}); got.Rule != nil {
		t.Errorf("non-alias model should fall through, matched %q", got.Rule.Name)
	}
}

// A rule must win over an alias of the same name: rules-first is the
// security invariant (a fail-closed pii rail intercepts before an alias).
func TestMatchRuleBeatsAlias(t *testing.T) {
	cfg := validConfig()
	cfg.Aliases = []Alias{{Name: "any", Backends: []string{"cloud-opus"}}}
	got := NewMatcher(cfg).Match(&RequestFeatures{Model: "any", Classification: "pii"})
	if got.Rule == nil || got.Rule.Name != "pii-stays-local" {
		t.Fatalf("expected pii rule to beat alias, got %v", got.Rule)
	}
	if !got.FailClosed {
		t.Error("pii rule should still report FailClosed")
	}
}

// TestMatchRuleInterceptsAliasByName pins the security contract that an alias
// is subject to rules exactly like a model name: a rule listing the alias in
// match.models intercepts it before alias resolution and routes to the rule's
// own backends. A refactor that resolved aliases ahead of rules would break
// this and reopen the sensitive-data gap.
func TestMatchRuleInterceptsAliasByName(t *testing.T) {
	cfg := validConfig()
	cfg.Aliases = []Alias{{Name: "frontier", Backends: []string{"cloud-opus"}}}
	cfg.Rules = []Rule{{
		Name:       "pii-stays-local",
		Match:      RuleMatch{DataClassification: []string{"pii"}, Models: []string{"frontier"}},
		Route:      RuleRoute{Backends: []string{"local-qwen"}},
		FailClosed: true,
	}}
	got := NewMatcher(cfg).Match(&RequestFeatures{Model: "frontier", Classification: "pii"})
	if got.Rule == nil || got.Rule.Name != "pii-stays-local" {
		t.Fatalf("rule naming the alias in match.models should intercept it, got %v", got.Rule)
	}
	if got.IsAlias {
		t.Error("a rule match must not be flagged as an alias route")
	}
	if !got.FailClosed {
		t.Error("the intercepting fail-closed rule should report FailClosed")
	}
	if len(got.Backends) != 1 || got.Backends[0] != "local-qwen" {
		t.Errorf("should route to the rule's local backend, not the alias's, got %v", got.Backends)
	}
}

// TestMatchRuleGlobInterceptsAlias pins the documented promise that a rule
// whose match.models glob covers an alias name intercepts it before alias
// resolution — the glob counterpart to TestMatchRuleInterceptsAliasByName.
func TestMatchRuleGlobInterceptsAlias(t *testing.T) {
	cfg := validConfig()
	cfg.Aliases = []Alias{{Name: "frontier", Backends: []string{"cloud-opus"}}}
	cfg.Rules = []Rule{{
		Name:       "pii-stays-local",
		Match:      RuleMatch{DataClassification: []string{"pii"}, Models: []string{"front*"}},
		Route:      RuleRoute{Backends: []string{"local-qwen"}},
		FailClosed: true,
	}}
	got := NewMatcher(cfg).Match(&RequestFeatures{Model: "frontier", Classification: "pii"})
	if got.Rule == nil || got.Rule.Name != "pii-stays-local" {
		t.Fatalf("glob rule should intercept the alias, got %v", got.Rule)
	}
	if got.IsAlias {
		t.Error("a rule match must not be flagged as an alias route")
	}
	if len(got.Backends) != 1 || got.Backends[0] != "local-qwen" {
		t.Errorf("should route to the rule's local backend, got %v", got.Backends)
	}
}
