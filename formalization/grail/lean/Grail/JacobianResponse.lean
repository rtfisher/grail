import Grail.Resolvent
import Mathlib.Analysis.Asymptotics.Lemmas
import Mathlib.Tactic.Ring

noncomputable section
open Filter Asymptotics
open scoped Topology
namespace Grail
variable {𝕜 E : Type*} [NontriviallyNormedField 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E]

/-- The resolvent substitution bound propagates through a common residual. -/
theorem jacobian_prediction_difference_le (Kt Ke Gt Ge : E →L[𝕜] E)
    (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) (b : E) :
    ‖Ge b - Gt b‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ * ‖b‖ := by
  calc
    _ = ‖(Ge - Gt) b‖ := rfl
    _ ≤ ‖Ge - Gt‖ * ‖b‖ := (Ge - Gt).le_opNorm b
    _ ≤ _ := mul_le_mul_of_nonneg_right
      (norm_resolvent_difference_le Kt Ke Gt Ge ht he) (norm_nonneg b)

/-- Pointwise second-order prediction cost with explicit uniform constants. -/
theorem jacobian_prediction_quadratic_bound (Kt Ke Gt Ge : E →L[𝕜] E)
    (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) (b : E)
    (η Me Mt D B : ℝ) (hD : 0 ≤ D) (hB : 0 ≤ B)
    (hGe : ‖Ge‖ ≤ Me) (hGt : ‖Gt‖ ≤ Mt)
    (hDiff : ‖Ke - Kt‖ ≤ D * |η|) (hb : ‖b‖ ≤ B * |η|) :
    ‖Ge b - Gt b‖ ≤ Me * D * Mt * B * η ^ 2 := by
  have hMe : 0 ≤ Me := (norm_nonneg Ge).trans hGe
  have hMt : 0 ≤ Mt := (norm_nonneg Gt).trans hGt
  have h12 : ‖Ge‖ * ‖Ke - Kt‖ ≤ Me * (D * |η|) :=
    mul_le_mul hGe hDiff (norm_nonneg _) hMe
  have h123 : ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ ≤ Me * (D * |η|) * Mt :=
    mul_le_mul h12 hGt (norm_nonneg _) (mul_nonneg hMe (mul_nonneg hD (abs_nonneg η)))
  calc
    _ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ * ‖b‖ :=
      jacobian_prediction_difference_le Kt Ke Gt Ge ht he b
    _ ≤ Me * (D * |η|) * Mt * (B * |η|) :=
      mul_le_mul h123 hb (norm_nonneg _) (mul_nonneg (mul_nonneg hMe
        (mul_nonneg hD (abs_nonneg η))) hMt)
    _ = Me * D * Mt * B * |η| ^ 2 := by ring
    _ = Me * D * Mt * B * η ^ 2 := by rw [sq_abs]

/-- Uniformly bounded resolvents and first-order Jacobian/residual errors
produce a genuine quadratic asymptotic prediction difference. -/
theorem jacobian_prediction_isBigO
    (Kt Ke Gt Ge : ℝ → E →L[𝕜] E) (b : ℝ → E)
    (Me Mt D B : ℝ) (hD : 0 ≤ D) (hB : 0 ≤ B)
    (h : ∀ᶠ η in 𝓝 (0 : ℝ),
      IsResolvent (Kt η) (Gt η) ∧ IsResolvent (Ke η) (Ge η) ∧
      ‖Ge η‖ ≤ Me ∧ ‖Gt η‖ ≤ Mt ∧
      ‖Ke η - Kt η‖ ≤ D * |η| ∧ ‖b η‖ ≤ B * |η|) :
    (fun η => Ge η (b η) - Gt η (b η)) =O[𝓝 (0 : ℝ)] (fun η => η ^ 2) := by
  refine IsBigO.of_bound (Me * D * Mt * B) ?_
  filter_upwards [h] with η hη
  have hbnd := jacobian_prediction_quadratic_bound (Kt η) (Ke η) (Gt η) (Ge η)
    hη.1 hη.2.1 (b η) η Me Mt D B hD hB hη.2.2.1 hη.2.2.2.1
    hη.2.2.2.2.1 hη.2.2.2.2.2
  simpa only [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg η)] using hbnd

/-- Complete manuscript-facing algebraic and propagated-error conclusions. -/
theorem jacobian_substitution_with_response (Kt Ke Gt Ge : E →L[𝕜] E)
    (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) :
    Ge - Gt = Ge * (Ke - Kt) * Gt ∧
    ‖Ge - Gt‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ ∧
    ∀ b : E, ‖Ge b - Gt b‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ * ‖b‖ :=
  ⟨resolvent_difference Kt Ke Gt Ge ht he,
    norm_resolvent_difference_le Kt Ke Gt Ge ht he,
    jacobian_prediction_difference_le Kt Ke Gt Ge ht he⟩

/-- All parts of the English substitution theorem, including its conditional
quadratic consequence for parameter-dependent Jacobian and residual families. -/
theorem jacobian_substitution_complete (Kt Ke Gt Ge : E →L[𝕜] E)
    (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) :
    (Ge - Gt = Ge * (Ke - Kt) * Gt ∧
      ‖Ge - Gt‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ ∧
      ∀ b : E, ‖Ge b - Gt b‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ * ‖b‖) ∧
    (∀ (Kt' Ke' Gt' Ge' : ℝ → E →L[𝕜] E) (b : ℝ → E)
      (Me Mt D B : ℝ), 0 ≤ D → 0 ≤ B →
      (∀ᶠ η in 𝓝 (0 : ℝ),
        IsResolvent (Kt' η) (Gt' η) ∧ IsResolvent (Ke' η) (Ge' η) ∧
        ‖Ge' η‖ ≤ Me ∧ ‖Gt' η‖ ≤ Mt ∧
        ‖Ke' η - Kt' η‖ ≤ D * |η| ∧ ‖b η‖ ≤ B * |η|) →
      (fun η => Ge' η (b η) - Gt' η (b η)) =O[𝓝 (0 : ℝ)]
        (fun η => η ^ 2)) := by
  exact ⟨jacobian_substitution_with_response Kt Ke Gt Ge ht he,
    fun Kt' Ke' Gt' Ge' b Me Mt D B hD hB h =>
      jacobian_prediction_isBigO Kt' Ke' Gt' Ge' b Me Mt D B hD hB h⟩

end Grail
