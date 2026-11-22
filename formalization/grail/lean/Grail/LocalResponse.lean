import Mathlib.Analysis.Calculus.ImplicitContDiff
import Mathlib.Analysis.Normed.Operator.NormedSpace
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel

/-!
# Local fixed-point response and nonlinear remainder

The formal state is a real Banach space; in the manuscript it is a finite
Euclidean space or its conservation-constrained tangent subspace.
-/

noncomputable section

open Filter Set Asymptotics
open scoped Topology ContDiff

namespace Grail

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- The implicit fixed-point equation for a perturbation of amplitude `η`. -/
def fixedPointResidual (T r : E → E) (p : ℝ × E) : E :=
  p.2 - T p.2 - p.1 • r p.2

lemma residual_contDiffAt {T r : E → E} {x₀ : E}
    (hT : ContDiffAt ℝ 2 T x₀) (hr : ContDiffAt ℝ 2 r x₀) :
    ContDiffAt ℝ 2 (fixedPointResidual T r) (0, x₀) := by
  exact (contDiffAt_snd.sub (hT.comp _ contDiffAt_snd)).sub
    (contDiffAt_fst.smul (hr.comp _ contDiffAt_snd))

lemma residual_state_partial {T r : E → E} {x₀ : E}
    (hT : ContDiffAt ℝ 2 T x₀) (hr : ContDiffAt ℝ 2 r x₀) :
    fderiv ℝ (fixedPointResidual T r) (0, x₀) ∘L
        ContinuousLinearMap.inr ℝ ℝ E =
      ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀ := by
  have hF := (residual_contDiffAt hT hr).differentiableAt (by norm_num)
  have hcomp := hF.hasFDerivAt.comp x₀ (ContinuousLinearMap.inr ℝ ℝ E).hasFDerivAt
  have hs : HasFDerivAt (fun x : E => x - T x)
      (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀) x₀ :=
    (hasFDerivAt_id x₀).sub (hT.differentiableAt (by norm_num)).hasFDerivAt
  have hcomp' : HasFDerivAt (fun x : E => x - T x)
      (fderiv ℝ (fixedPointResidual T r) (0, x₀) ∘L
        ContinuousLinearMap.inr ℝ ℝ E) x₀ := by
    simpa [fixedPointResidual, Function.comp_def] using hcomp
  exact hcomp'.unique hs

lemma residual_parameter_partial {T r : E → E} {x₀ : E}
    (hT : ContDiffAt ℝ 2 T x₀) (hr : ContDiffAt ℝ 2 r x₀)
    (hfixed : T x₀ = x₀) :
    fderiv ℝ (fixedPointResidual T r) (0, x₀) ∘L
        ContinuousLinearMap.inl ℝ ℝ E =
      -(ContinuousLinearMap.id ℝ ℝ).smulRight (r x₀) := by
  have hF := (residual_contDiffAt hT hr).differentiableAt (by norm_num)
  have hp : HasFDerivAt (fun η : ℝ => (η, x₀))
      (ContinuousLinearMap.inl ℝ ℝ E) 0 := by
    simpa only [Function.id_def, ContinuousLinearMap.inl] using
      (hasFDerivAt_id (0 : ℝ)).prodMk (hasFDerivAt_const x₀ 0)
  have hcomp := hF.hasFDerivAt.comp 0 hp
  have hs : HasFDerivAt (fun η : ℝ => -(η • r x₀))
      (-(ContinuousLinearMap.id ℝ ℝ).smulRight (r x₀)) 0 := by
    convert (-(ContinuousLinearMap.id ℝ ℝ).smulRight (r x₀)).hasFDerivAt (x := 0) using 1
    funext η
    rfl
  have hcomp' : HasFDerivAt (fun η : ℝ => -(η • r x₀))
      (fderiv ℝ (fixedPointResidual T r) (0, x₀) ∘L
        ContinuousLinearMap.inl ℝ ℝ E) 0 := by
    simpa [fixedPointResidual, Function.comp_def, hfixed] using hcomp
  exact hcomp'.unique hs

/-- A small smooth surrogate perturbation has a locally unique smooth
fixed-point branch, whose derivative is the resolvent response. -/
theorem local_fixed_point_response {T r : E → E} {x₀ : E}
    (hT : ContDiffAt ℝ 2 T x₀) (hr : ContDiffAt ℝ 2 r x₀)
    (hfixed : T x₀ = x₀)
    (hregular : (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).IsInvertible) :
    ∃ ψ : ℝ → E,
      ψ 0 = x₀ ∧
      ContDiffAt ℝ 2 ψ 0 ∧
      (∀ᶠ η in 𝓝 (0 : ℝ), ψ η = T (ψ η) + η • r (ψ η)) ∧
      (∀ᶠ p in 𝓝 ((0 : ℝ), x₀),
        p.2 = T p.2 + p.1 • r p.2 ↔ ψ p.1 = p.2) ∧
      HasDerivAt ψ
        ((ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀)) 0 := by
  have hF := residual_contDiffAt hT hr
  have hinv : (fderiv ℝ (fixedPointResidual T r) (0, x₀) ∘L
      ContinuousLinearMap.inr ℝ ℝ E).IsInvertible := by
    rwa [residual_state_partial hT hr]
  let ψ := hF.implicitFunction (by norm_num) hinv
  refine ⟨ψ, hF.implicitFunction_apply_self (by norm_num) hinv,
    hF.contDiffAt_implicitFunction (by norm_num) hinv, ?_, ?_, ?_⟩
  · filter_upwards [hF.eventually_apply_implicitFunction (by norm_num) hinv] with η hη
    simpa [ψ, fixedPointResidual, hfixed, sub_eq_zero, sub_eq_iff_eq_add, add_comm] using hη
  · filter_upwards [hF.eventually_apply_eq_iff_implicitFunction (by norm_num) hinv]
      with p hp
    simpa [ψ, fixedPointResidual, hfixed, sub_eq_zero, sub_eq_iff_eq_add, add_comm] using hp
  · have hd := (hF.hasStrictFDerivAt_implicitFunction (by norm_num) hinv).hasFDerivAt
    rw [residual_state_partial hT hr, residual_parameter_partial hT hr hfixed] at hd
    simpa [ψ, ContinuousLinearMap.comp_apply] using hd.hasDerivAt


section TaylorBound

variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- A bounded second derivative gives the sharp quadratic Taylor estimate
along a segment. No Taylor-remainder hypothesis is assumed. -/
theorem norm_taylor_remainder_le {T : E → F} {s : Set E} {x₀ x : E} {M : ℝ}
    (hs : Convex ℝ s) (hx₀ : x₀ ∈ s) (hx : x ∈ s)
    (hT : ∀ y ∈ s, ContDiffAt ℝ 2 T y)
    (hM : ∀ y ∈ s, ‖fderiv ℝ (fderiv ℝ T) y‖ ≤ M) :
    ‖T x - T x₀ - fderiv ℝ T x₀ (x - x₀)‖ ≤ M / 2 * ‖x - x₀‖ ^ 2 := by
  let v := x - x₀
  let R : ℝ → F := fun t => T (x₀ + t • v) - T x₀ - t • fderiv ℝ T x₀ v
  have hmem (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) : x₀ + t • v ∈ s :=
    hs.add_smul_sub_mem hx₀ hx ht
  have hR (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
      HasDerivAt R (fderiv ℝ T (x₀ + t • v) v - fderiv ℝ T x₀ v) t := by
    have hline : HasDerivAt (fun u : ℝ => x₀ + u • v) v t := by
      simpa only [Function.id_def, one_smul] using
        ((hasDerivAt_id t).smul_const v).const_add x₀
    have hlin : HasDerivAt (fun u : ℝ => u • fderiv ℝ T x₀ v)
        (fderiv ℝ T x₀ v) t := by
      simpa only [Function.id_def, one_smul] using
        (hasDerivAt_id t).smul_const (fderiv ℝ T x₀ v)
    exact (((hT _ (hmem t ht)).differentiableAt (by norm_num)).hasFDerivAt.comp_hasDerivAt
      t hline).sub_const (T x₀) |>.sub hlin
  have hD (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
      ‖fderiv ℝ T (x₀ + t • v) v - fderiv ℝ T x₀ v‖ ≤ M * ‖v‖ ^ 2 * t := by
    have hLip : ‖fderiv ℝ T (x₀ + t • v) - fderiv ℝ T x₀‖ ≤
        M * ‖(x₀ + t • v) - x₀‖ :=
      hs.norm_image_sub_le_of_norm_fderiv_le
        (fun y hy => ((hT y hy).fderiv_right (m := 1) (by norm_num)).differentiableAt
          (by norm_num)) hM hx₀ (hmem t ht)
    calc
      _ = ‖(fderiv ℝ T (x₀ + t • v) - fderiv ℝ T x₀) v‖ := rfl
      _ ≤ ‖fderiv ℝ T (x₀ + t • v) - fderiv ℝ T x₀‖ * ‖v‖ :=
        ContinuousLinearMap.le_opNorm _ _
      _ ≤ (M * ‖(x₀ + t • v) - x₀‖) * ‖v‖ :=
        mul_le_mul_of_nonneg_right hLip (norm_nonneg _)
      _ = M * ‖v‖ ^ 2 * t := by
        simp only [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1]
        ring
  have hB (t : ℝ) : HasDerivAt (fun u : ℝ => (M / 2 * ‖v‖ ^ 2) * u ^ 2)
      (M * ‖v‖ ^ 2 * t) t := by
    have hd := ((hasDerivAt_id t).pow 2).const_mul (M / 2 * ‖v‖ ^ 2)
    simp only [Function.id_def, Pi.pow_apply] at hd
    convert hd using 1
    ring
  have hb := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (fun t ht => (hR t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hR t ⟨ht.1, ht.2.le⟩).hasDerivWithinAt)
    (by simp [R]) hB (fun t ht => hD t ⟨ht.1, ht.2.le⟩)
    (by norm_num : (1 : ℝ) ∈ Icc (0 : ℝ) 1)
  simpa [R, v] using hb

end TaylorBound

/-- The actual fixed-point prediction error is controlled by the reference
cycle's Hessian and the residual's derivative on the connecting segment. -/
theorem nonlinear_response_remainder_bound {T r : E → E} {s : Set E}
    {x₀ x : E} {η M L : ℝ}
    (hs : Convex ℝ s) (hx₀ : x₀ ∈ s) (hx : x ∈ s)
    (hT : ∀ y ∈ s, ContDiffAt ℝ 2 T y)
    (hM : ∀ y ∈ s, ‖fderiv ℝ (fderiv ℝ T) y‖ ≤ M)
    (hr : ∀ y ∈ s, DifferentiableAt ℝ r y)
    (hL : ∀ y ∈ s, ‖fderiv ℝ r y‖ ≤ L)
    (hfixed : T x₀ = x₀) (hperturbed : x = T x + η • r x)
    (hregular : (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).IsInvertible) :
    ‖(x - x₀) - η • (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀)‖ ≤
      ‖(ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse‖ *
        (M / 2 * ‖x - x₀‖ ^ 2 + |η| * L * ‖x - x₀‖) := by
  let K := fderiv ℝ T x₀
  let A := ContinuousLinearMap.id ℝ E - K
  let G := A.inverse
  let e := x - x₀
  let R := T x - T x₀ - K e
  have he : e = (T x - T x₀) + η • r x := by
    change x - x₀ = (T x - T x₀) + η • r x
    calc
      _ = (T x + η • r x) - x₀ := congrArg (fun y => y - x₀) hperturbed
      _ = _ := by rw [hfixed]; abel
  have hA : A e = R + η • r x := by
    change e - K e = (T x - T x₀ - K e) + η • r x
    calc
      _ = ((T x - T x₀) + η • r x) - K e := congrArg (fun y => y - K e) he
      _ = _ := by abel
  have hG : e = G (R + η • r x) := by
    exact (hregular.inverse_apply_self e).symm.trans (congrArg G hA)
  have herror : e - η • G (r x₀) = G (R + η • (r x - r x₀)) := by
    calc
      _ = G (R + η • r x) - η • G (r x₀) := congrArg (fun y => y - η • G (r x₀)) hG
      _ = _ := by simp only [map_add, map_sub, map_smul, smul_sub]; abel
  have hRt : ‖R‖ ≤ M / 2 * ‖e‖ ^ 2 :=
    norm_taylor_remainder_le hs hx₀ hx hT hM
  have hrr : ‖r x - r x₀‖ ≤ L * ‖e‖ :=
    hs.norm_image_sub_le_of_norm_fderiv_le hr hL hx₀ hx
  change ‖e - η • G (r x₀)‖ ≤ ‖G‖ * (M / 2 * ‖e‖ ^ 2 + |η| * L * ‖e‖)
  rw [herror]
  calc
    _ ≤ ‖G‖ * ‖R + η • (r x - r x₀)‖ := G.le_opNorm _
    _ ≤ ‖G‖ * (‖R‖ + |η| * ‖r x - r x₀‖) := by
      apply mul_le_mul_of_nonneg_left _ (norm_nonneg G)
      simpa [norm_smul, Real.norm_eq_abs] using norm_add_le R (η • (r x - r x₀))
    _ ≤ ‖G‖ * (M / 2 * ‖e‖ ^ 2 + |η| * (L * ‖e‖)) :=
      mul_le_mul_of_nonneg_left
        (add_le_add hRt (mul_le_mul_of_nonneg_left hrr (abs_nonneg η))) (norm_nonneg G)
    _ = _ := by ring


/-- A continuous real-linear operator on the scalar field has norm equal
  to the norm of its value at one. -/
lemma norm_scalar_input_operator (A : ℝ →L[ℝ] E) : ‖A‖ = ‖A 1‖ := by
  apply le_antisymm
  · apply A.opNorm_le_bound (norm_nonneg _)
    intro x
    have hAx : A x = x • A 1 := by
      simpa only [smul_eq_mul, mul_one] using A.map_smul x (1 : ℝ)
    rw [hAx, norm_smul, Real.norm_eq_abs, mul_comm]
  · simpa using A.le_opNorm (1 : ℝ)

/-- Every C2 curve has a genuinely quadratic first-order Taylor remainder. -/
theorem smooth_curve_quadratic_remainder {ψ : ℝ → E}
    (hψ : ContDiffAt ℝ 2 ψ 0) :
    (fun η : ℝ => ψ η - ψ 0 - η • deriv ψ 0) =O[𝓝 (0 : ℝ)]
      (fun η : ℝ => η ^ 2) := by
  let M := ‖fderiv ℝ (fderiv ℝ ψ) 0‖ + 1
  have hcont : ContinuousAt (fderiv ℝ (fderiv ℝ ψ)) 0 :=
    ((hψ.fderiv_right (m := 1) (by norm_num)).fderiv_right (m := 0)
      (by norm_num)).continuousAt
  have hnorm : ContinuousAt (fun η : ℝ => ‖fderiv ℝ (fderiv ℝ ψ) η‖) 0 := by
    have heval : ContinuousAt (fun η : ℝ => fderiv ℝ (fderiv ℝ ψ) η 1 1) 0 :=
      (continuous_eval_const (1 : ℝ)).continuousAt.comp
        ((continuous_eval_const (1 : ℝ)).continuousAt.comp hcont)
    simpa only [norm_scalar_input_operator] using heval.norm
  have hlt : ‖fderiv ℝ (fderiv ℝ ψ) 0‖ < M := lt_add_one _
  have hbound' : ∀ᶠ η : ℝ in 𝓝 0, ‖fderiv ℝ (fderiv ℝ ψ) η‖ < M :=
    hnorm.eventually (Iio_mem_nhds hlt)
  have hbound : ∀ᶠ η : ℝ in 𝓝 0, ‖fderiv ℝ (fderiv ℝ ψ) η‖ ≤ M :=
    hbound'.mono (fun _ h => h.le)
  obtain ⟨ε, hε, heps⟩ := Metric.mem_nhds_iff.mp
    ((hψ.eventually (by norm_num)).and hbound)
  refine IsBigO.of_bound (M / 2) ?_
  filter_upwards [Metric.ball_mem_nhds (0 : ℝ) hε] with η hη
  have ht := norm_taylor_remainder_le (T := ψ) (x₀ := (0 : ℝ)) (x := η) (M := M)
    (convex_ball (0 : ℝ) ε)
    (by simpa using hε) hη
    (fun y hy => (heps hy).1) (fun y hy => (heps hy).2)
  simpa [fderiv_eq_smul_deriv, norm_pow] using ht

/-- Full local-response statement: existence, local uniqueness, C2 regularity,
resolvent derivative, and quadratic asymptotic remainder. -/
theorem local_fixed_point_response_with_remainder {T r : E → E} {x₀ : E}
    (hT : ContDiffAt ℝ 2 T x₀) (hr : ContDiffAt ℝ 2 r x₀)
    (hfixed : T x₀ = x₀)
    (hregular : (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).IsInvertible) :
    ∃ ψ : ℝ → E,
      ψ 0 = x₀ ∧
      ContDiffAt ℝ 2 ψ 0 ∧
      (∀ᶠ η in 𝓝 (0 : ℝ), ψ η = T (ψ η) + η • r (ψ η)) ∧
      (∀ᶠ p in 𝓝 ((0 : ℝ), x₀),
        p.2 = T p.2 + p.1 • r p.2 ↔ ψ p.1 = p.2) ∧
      HasDerivAt ψ
        ((ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀)) 0 ∧
      ((fun η : ℝ => ψ η - x₀ -
        η • (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀))
          =O[𝓝 (0 : ℝ)] (fun η : ℝ => η ^ 2)) := by
  obtain ⟨ψ, h0, hC2, hfix, huniq, hd⟩ := local_fixed_point_response hT hr hfixed hregular
  refine ⟨ψ, h0, hC2, hfix, huniq, hd, ?_⟩
  simpa [h0, hd.deriv] using smooth_curve_quadratic_remainder hC2


/-- Full derivative-controlled nonlinear result on a fixed convex neighborhood:
the branch is constructed by IFT, the explicit error estimate is derived from
derivative bounds, and the same branch has an O(eta^2) remainder. -/
theorem local_response_with_explicit_remainder {T r : E → E} {s : Set E}
    {x₀ : E} {M L : ℝ} (hs : Convex ℝ s) (hsn : s ∈ 𝓝 x₀)
    (hT : ∀ y ∈ s, ContDiffAt ℝ 2 T y)
    (hr : ∀ y ∈ s, ContDiffAt ℝ 2 r y)
    (hM : ∀ y ∈ s, ‖fderiv ℝ (fderiv ℝ T) y‖ ≤ M)
    (hL : ∀ y ∈ s, ‖fderiv ℝ r y‖ ≤ L)
    (hfixed : T x₀ = x₀)
    (hregular : (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).IsInvertible) :
    ∃ ψ : ℝ → E,
      ψ 0 = x₀ ∧ ContDiffAt ℝ 2 ψ 0 ∧
      (∀ᶠ η in 𝓝 (0 : ℝ), ψ η = T (ψ η) + η • r (ψ η)) ∧
      (∀ᶠ p in 𝓝 ((0 : ℝ), x₀),
        p.2 = T p.2 + p.1 • r p.2 ↔ ψ p.1 = p.2) ∧
      HasDerivAt ψ
        ((ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀)) 0 ∧
      (∀ᶠ η in 𝓝 (0 : ℝ),
        ‖(ψ η - x₀) - η • (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀)‖ ≤
          ‖(ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse‖ *
            (M / 2 * ‖ψ η - x₀‖ ^ 2 + |η| * L * ‖ψ η - x₀‖)) ∧
      ((fun η : ℝ => ψ η - x₀ -
        η • (ContinuousLinearMap.id ℝ E - fderiv ℝ T x₀).inverse (r x₀))
          =O[𝓝 (0 : ℝ)] (fun η : ℝ => η ^ 2)) := by
  have hx₀ : x₀ ∈ s := mem_of_mem_nhds hsn
  obtain ⟨ψ, h0, hC2, hfix, huniq, hd, hO⟩ :=
    local_fixed_point_response_with_remainder (hT x₀ hx₀) (hr x₀ hx₀) hfixed hregular
  refine ⟨ψ, h0, hC2, hfix, huniq, hd, ?_, hO⟩
  have hstay : ∀ᶠ η in 𝓝 (0 : ℝ), ψ η ∈ s :=
    hC2.continuousAt.eventually (by simpa [h0] using hsn)
  filter_upwards [hfix, hstay] with η hη hmem
  exact nonlinear_response_remainder_bound hs hx₀ hmem hT hM
    (fun y hy => (hr y hy).differentiableAt (by norm_num)) hL hfixed hη hregular

end Grail
