/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	inferencev1alpha1 "github.com/defilantech/llmkube/api/v1alpha1"
)

// validGatewayRouter builds a baseline dataPlane: Gateway ModelRouter that
// passes both the static spec validation and every gateway-mode honest-boundary
// check. Each test mutates one field to exercise a single rejection branch.
func validGatewayRouter() *inferencev1alpha1.ModelRouter {
	return &inferencev1alpha1.ModelRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "qwen-router", Namespace: "default"},
		Spec: inferencev1alpha1.ModelRouterSpec{
			DataPlane: inferencev1alpha1.ModelRouterDataPlaneGateway,
			GatewayRef: &inferencev1alpha1.GatewayReference{
				Name:      "ai-gateway",
				Namespace: "ai-gateway",
			},
			Backends: []inferencev1alpha1.RouterBackend{
				{Name: "qwen-cuda", InferenceServiceRef: corev1LocalRef("qwen-cuda")},
				{Name: "qwen-metal", InferenceServiceRef: corev1LocalRef("qwen-metal")},
			},
			Rules: []inferencev1alpha1.RouterRule{
				{
					Name:  "qwen",
					Match: &inferencev1alpha1.RuleMatch{Models: []string{"qwen35-27b"}},
					Route: inferencev1alpha1.RuleRoute{
						Backends: []string{"qwen-cuda", "qwen-metal"},
						Strategy: "primary-fallback",
					},
				},
			},
		},
	}
}

func TestModelRouterValidator_Create(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	tests := []struct {
		name      string
		mutate    func(*inferencev1alpha1.ModelRouter)
		wantError bool
	}{
		{
			name:      "valid gateway router accepted",
			mutate:    func(*inferencev1alpha1.ModelRouter) {},
			wantError: false,
		},
		{
			name: "static spec violation (undefined defaultRoute) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.DefaultRoute = "does-not-exist"
			},
			wantError: true,
		},
		{
			name: "unsupported match (taskComplexity) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Rules[0].Match.TaskComplexity = "high"
			},
			wantError: true,
		},
		{
			name: "unsupported strategy (shadow) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Rules[0].Route.Strategy = "shadow"
			},
			wantError: true,
		},
		{
			name: "unsupported budget (maxUSD dollar budget) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Policy = &inferencev1alpha1.RouterPolicy{
					Budgets: []inferencev1alpha1.BudgetSpec{
						{Name: "spend", Scope: "router", MaxUSD: "10.50"},
					},
				}
			},
			wantError: true,
		},
		{
			name: "unsupported budget scope (rule) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				maxTokens := int64(1000)
				mr.Spec.Policy = &inferencev1alpha1.RouterPolicy{
					Budgets: []inferencev1alpha1.BudgetSpec{
						{Name: "per-rule", Scope: "rule", RuleName: "qwen", MaxTokens: &maxTokens},
					},
				}
			},
			wantError: true,
		},
		{
			name: "invalid auth (jwt missing issuer/jwksURI/teamClaim) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Policy = &inferencev1alpha1.RouterPolicy{
					Auth: &inferencev1alpha1.RouterAuthSpec{
						JWT: &inferencev1alpha1.JWTAuthSpec{Provider: "keycloak"},
					},
				}
			},
			wantError: true,
		},
		{
			name: "invalid authorization (allowlists without jwt) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Policy = &inferencev1alpha1.RouterPolicy{
					Auth: &inferencev1alpha1.RouterAuthSpec{
						Allowlists: []inferencev1alpha1.TeamModelAllowlist{
							{Team: "platform", Models: []string{"qwen35-27b"}},
						},
					},
				}
			},
			wantError: true,
		},
		{
			name: "auditLog in gateway mode rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Policy = &inferencev1alpha1.RouterPolicy{
					AuditLog: &inferencev1alpha1.AuditLogPolicy{Sink: "stdout"},
				}
			},
			wantError: true,
		},
		{
			name: "unsafe sensitive route (pii rule not failClosed) rejected",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Rules[0].Match.DataClassification = []string{"pii"}
				mr.Spec.Rules[0].FailClosed = false
			},
			wantError: true,
		},
		{
			name: "safe sensitive route (pii rule failClosed, local-only) accepted",
			mutate: func(mr *inferencev1alpha1.ModelRouter) {
				mr.Spec.Rules[0].Match.DataClassification = []string{"pii"}
				mr.Spec.Rules[0].FailClosed = true
			},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mr := validGatewayRouter()
			tt.mutate(mr)
			_, err := v.ValidateCreate(ctx, mr)
			if tt.wantError && err == nil {
				t.Fatalf("expected validation error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
			if tt.wantError && err != nil && !apierrors.IsInvalid(err) {
				t.Fatalf("expected an Invalid error, got %T: %v", err, err)
			}
		})
	}
}

// TestModelRouterValidator_ProxyModeSkipsGatewayChecks proves the gateway-mode
// honest-boundary checks are NOT applied to a Proxy-mode router: a config that
// the gateway data plane cannot express (a dollar budget, an auditLog directive)
// is perfectly valid in Proxy mode, where the router-proxy enforces it directly.
func TestModelRouterValidator_ProxyModeSkipsGatewayChecks(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	mr := validGatewayRouter()
	// Proxy mode (the default). The router-proxy honors dollar budgets and
	// auditLog, so neither is a violation here.
	mr.Spec.DataPlane = inferencev1alpha1.ModelRouterDataPlaneProxy
	mr.Spec.GatewayRef = nil
	mr.Spec.Policy = &inferencev1alpha1.RouterPolicy{
		Budgets: []inferencev1alpha1.BudgetSpec{
			{Name: "spend", Scope: "router", MaxUSD: "10.50"},
		},
		AuditLog: &inferencev1alpha1.AuditLogPolicy{Sink: "stdout"},
	}
	// A taskComplexity match is inexpressible in the gateway data plane but is
	// a first-class Proxy-mode match.
	mr.Spec.Rules[0].Match.TaskComplexity = "high"

	if _, err := v.ValidateCreate(ctx, mr); err != nil {
		t.Fatalf("expected Proxy-mode router with gateway-inexpressible config to pass, got %v", err)
	}
}

// TestModelRouterValidator_ProxyModeStillRunsStaticChecks proves a Proxy-mode
// router is still subject to the static spec validation (which is data-plane
// independent): an undefined defaultRoute is rejected regardless of data plane.
func TestModelRouterValidator_ProxyModeStillRunsStaticChecks(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	mr := validGatewayRouter()
	mr.Spec.DataPlane = inferencev1alpha1.ModelRouterDataPlaneProxy
	mr.Spec.GatewayRef = nil
	mr.Spec.DefaultRoute = "does-not-exist"

	_, err := v.ValidateCreate(ctx, mr)
	if err == nil {
		t.Fatalf("expected static validation to reject undefined defaultRoute in Proxy mode")
	}
	if !apierrors.IsInvalid(err) {
		t.Fatalf("expected an Invalid error, got %T: %v", err, err)
	}
}

// TestModelRouterValidator_AliasBackendCollisionWarns proves an alias whose
// name shadows a backend's /v1/models id is accepted (no error) but surfaces
// an apply-time warning, that the collision is detected against the published
// id (external Model, not just Name), and that a non-colliding alias warns
// nothing.
func TestModelRouterValidator_AliasBackendCollisionWarns(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	// A cloud backend publishes under its external Model ("gpt-4o"), not its
	// Name ("openai-primary"): an alias named after the model id must warn.
	withCloud := func() *inferencev1alpha1.ModelRouter {
		mr := validGatewayRouter()
		mr.Spec.Backends = append(mr.Spec.Backends, inferencev1alpha1.RouterBackend{
			Name: "openai-primary",
			External: &inferencev1alpha1.ExternalProvider{
				Provider: "openai",
				Model:    "gpt-4o",
			},
			Tier: testRouterTierCloud,
		})
		return mr
	}

	cases := []struct {
		name      string
		alias     string
		wantWarns int
	}{
		{name: "collides with backend Name", alias: "qwen-cuda", wantWarns: 1},
		{name: "collides with backend Model id", alias: "gpt-4o", wantWarns: 1},
		{name: "no collision", alias: "fast", wantWarns: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mr := withCloud()
			mr.Spec.Aliases = []inferencev1alpha1.RouterAlias{
				{Name: tc.alias, Backends: []string{"qwen-cuda"}},
			}
			warnings, err := v.ValidateCreate(ctx, mr)
			if err != nil {
				t.Fatalf("collision is a warning, not an error; got %v", err)
			}
			if len(warnings) != tc.wantWarns {
				t.Errorf("expected %d warning(s), got %d: %v", tc.wantWarns, len(warnings), warnings)
			}
		})
	}
}

// TestModelRouterValidator_Update reuses the create branches when the spec
// changes: a spec-changing update applies the same invariants.
func TestModelRouterValidator_Update(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	// A spec-changing update into an invalid gateway-mode state is rejected.
	bad := validGatewayRouter()
	bad.Spec.Rules[0].Route.Strategy = "shadow"
	if _, err := v.ValidateUpdate(ctx, validGatewayRouter(), bad); err == nil {
		t.Fatalf("expected spec-changing update into an unsupported strategy to fail")
	}

	// A spec-changing update to a still-valid state passes.
	good := validGatewayRouter()
	good.Spec.Rules[0].Match.Models = []string{"qwen35-27b", "qwen35-27b-instruct"}
	if _, err := v.ValidateUpdate(ctx, validGatewayRouter(), good); err != nil {
		t.Fatalf("expected spec-changing update to a valid spec to pass, got %v", err)
	}
}

// TestModelRouterValidator_Update_Grandfathering proves that a status/metadata
// patch leaving the spec untouched is accepted even when the spec is already
// invalid, so a grandfathered bad ModelRouter cannot be wedged (the gateway
// reconciler patches status every reconcile). A patch that actually changes the
// invalid spec is still rejected.
func TestModelRouterValidator_Update_Grandfathering(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	// A router that would fail create-time validation (unsupported strategy)
	// is presumed to predate the webhook.
	invalid := validGatewayRouter()
	invalid.Spec.Rules[0].Route.Strategy = "shadow"
	if _, err := v.ValidateCreate(ctx, invalid.DeepCopy()); err == nil {
		t.Fatalf("test setup: expected the invalid router to fail create validation")
	}

	// Status/metadata-only patch: spec unchanged -> accepted.
	statusPatched := invalid.DeepCopy()
	statusPatched.Labels = map[string]string{"patched": "true"}
	statusPatched.Status.Phase = "Degraded"
	if _, err := v.ValidateUpdate(ctx, invalid.DeepCopy(), statusPatched); err != nil {
		t.Fatalf("expected status-only update of a spec-invalid router to be accepted (grandfathering), got %v", err)
	}

	// Spec-changing patch into a still-invalid state -> rejected.
	specPatched := invalid.DeepCopy()
	specPatched.Spec.Rules[0].Match.TaskComplexity = "high"
	if _, err := v.ValidateUpdate(ctx, invalid.DeepCopy(), specPatched); err == nil {
		t.Fatalf("expected a spec-changing update of an invalid router to be re-validated and rejected")
	}
}

// TestModelRouterValidator_Delete proves delete is always allowed.
func TestModelRouterValidator_Delete(t *testing.T) {
	v := &ModelRouterValidator{}
	ctx := context.Background()

	// Even a thoroughly invalid router deletes cleanly.
	mr := validGatewayRouter()
	mr.Spec.Rules[0].Route.Strategy = "shadow"
	if _, err := v.ValidateDelete(ctx, mr); err != nil {
		t.Fatalf("expected ValidateDelete to always allow, got %v", err)
	}
}
