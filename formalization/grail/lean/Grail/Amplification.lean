import Mathlib.Analysis.Normed.Operator.NNNorm
import Mathlib.Analysis.Normed.Operator.NormedSpace
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.Analysis.CStarAlgebra.Spectrum
import Mathlib.Analysis.CStarAlgebra.ContinuousLinearMap
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.LinearAlgebra.Eigenspace.Matrix
import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Basic
import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Instances
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# Worst-direction amplification

The minimum singular value is used in its independent variational form,
`inf (‖A x‖ / ‖x‖)` over nonzero vectors. For finite-dimensional Euclidean
spaces this is the usual least singular value. This definition does not
refer to inverse norms; the inverse-norm identity is proved below.
-/

noncomputable section
open Set
open scoped ENNReal NNReal

namespace Grail

section MinimumGain
variable {𝕜 E F : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [NormedAddCommGroup F]
  [NormedSpace 𝕜 E] [NormedSpace 𝕜 F]

/-- Variational least gain; in Euclidean space, the least singular value. -/
def minimumGain (A : E →L[𝕜] F) : ℝ :=
  sInf ((fun x : E ↦ ‖A x‖ / ‖x‖) '' {x : E | x ≠ 0})

lemma gain_ratios_bddBelow (A : E →L[𝕜] F) :
    BddBelow ((fun x : E ↦ ‖A x‖ / ‖x‖) '' {x : E | x ≠ 0}) := by
  refine ⟨0, ?_⟩
  rintro y ⟨x, hx, rfl⟩
  positivity

lemma gain_ratios_nonempty [Nontrivial E] (A : E →L[𝕜] F) :
    (((fun x : E ↦ ‖A x‖ / ‖x‖) '' {x : E | x ≠ 0})).Nonempty := by
  obtain ⟨x, hx⟩ := exists_ne (0 : E)
  exact ⟨_, x, hx, rfl⟩

/-- The least gain bounds the response in every direction. -/
theorem minimumGain_mul_norm_le (A : E →L[𝕜] F) (x : E) :
    minimumGain A * ‖x‖ ≤ ‖A x‖ := by
  by_cases hx : x = 0
  · simp [hx]
  · have h : minimumGain A ≤ ‖A x‖ / ‖x‖ :=
      csInf_le (gain_ratios_bddBelow A) ⟨x, hx, rfl⟩
    exact (le_div_iff₀ (norm_pos_iff.mpr hx)).mp h

/-- A bounded inverse gives a strictly positive lower bound on least gain. -/
theorem inv_norm_symm_le_minimumGain [Nontrivial E] (A : E ≃L[𝕜] F) :
    1 / ‖(A.symm : F →L[𝕜] E)‖ ≤ minimumGain (A : E →L[𝕜] F) := by
  apply le_csInf (gain_ratios_nonempty (A : E →L[𝕜] F))
  rintro y ⟨x, hx, rfl⟩
  have h := (A.symm : F →L[𝕜] E).le_opNorm (A x)
  simp only [ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.symm_apply_apply] at h
  apply (div_le_div_iff₀ A.norm_symm_pos (norm_pos_iff.mpr hx)).mpr
  simpa [mul_comm] using h

/-- Theorem 5 in variational form: maximum inverse amplification is the
reciprocal of the independent minimum forward gain. -/
theorem minimumGain_eq_inv_norm_symm [Nontrivial E] (A : E ≃L[𝕜] F) :
    minimumGain (A : E →L[𝕜] F) = 1 / ‖(A.symm : F →L[𝕜] E)‖ := by
  have hlo := inv_norm_symm_le_minimumGain A
  have hp : 0 < minimumGain (A : E →L[𝕜] F) :=
    lt_of_lt_of_le (one_div_pos.mpr A.norm_symm_pos) hlo
  have hb : ‖(A.symm : F →L[𝕜] E)‖ ≤ 1 / minimumGain (A : E →L[𝕜] F) := by
    apply (A.symm : F →L[𝕜] E).opNorm_le_bound (by positivity)
    intro y
    have h := minimumGain_mul_norm_le (A : E →L[𝕜] F) (A.symm y)
    simp only [ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.apply_symm_apply] at h
    have h' : ‖A.symm y‖ ≤ ‖y‖ / minimumGain (A : E →L[𝕜] F) :=
      (le_div_iff₀ hp).mpr (by simpa [mul_comm] using h)
    simpa [div_eq_mul_inv, mul_comm] using h'
  apply le_antisymm ?_ hlo
  apply (le_div_iff₀ A.norm_symm_pos).mpr
  have h := (le_div_iff₀ hp).mp hb
  simpa [mul_comm] using h

/-- Inverse operator norm equals reciprocal least gain. -/
theorem norm_symm_eq_inv_minimumGain [Nontrivial E] (A : E ≃L[𝕜] F) :
    ‖(A.symm : F →L[𝕜] E)‖ = 1 / minimumGain (A : E →L[𝕜] F) := by
  rw [minimumGain_eq_inv_norm_symm]
  simp

/-- Every directional response is bounded by the operator gain. -/
theorem directional_gain_le (G : E →L[𝕜] F) {b : E} (hb : b ≠ 0) :
    ‖G b‖ / ‖b‖ ≤ ‖G‖ := by
  exact (div_le_iff₀ (norm_pos_iff.mpr hb)).mpr (G.le_opNorm b)

/-- The supremum of unit-direction gains is exactly the operator norm. -/
theorem worst_unit_gain_eq_norm (G : E →L[𝕜] F) :
    sSup ((fun b : E ↦ ‖G b‖) '' Metric.sphere 0 1) = ‖G‖ :=
  G.sSup_sphere_eq_norm


/-- Normalizing a nonzero vector leaves its stretch factor unchanged. -/
theorem gain_ratio_image_eq_unit_image (A : E →L[𝕜] F) :
    ((fun x : E ↦ ‖A x‖ / ‖x‖) '' {x : E | x ≠ 0}) =
      ((fun x : E ↦ ‖A x‖) '' Metric.sphere 0 1) := by
  ext y
  constructor
  · rintro ⟨x, hx, rfl⟩
    let v : E := ((‖x‖ : 𝕜)⁻¹) • x
    have hxnorm : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
    have hv : ‖v‖ = 1 := by
      simp [v, norm_smul, norm_inv, hxnorm]
    refine ⟨v, mem_sphere_zero_iff_norm.mpr hv, ?_⟩
    simp [v, map_smul, norm_smul, norm_inv, div_eq_mul_inv, mul_comm]
  · rintro ⟨x, hx, rfl⟩
    have hxn : ‖x‖ = 1 := mem_sphere_zero_iff_norm.mp hx
    have hx0 : x ≠ 0 := by
      intro h
      simp [h] at hxn
    exact ⟨x, hx0, by simp [hxn]⟩

/-- The ratio definition agrees exactly with the unit-sphere definition. -/
theorem minimumGain_eq_unitSphere (A : E →L[𝕜] F) :
    minimumGain A = sInf ((fun x : E ↦ ‖A x‖) '' Metric.sphere 0 1) := by
  rw [minimumGain, gain_ratio_image_eq_unit_image]

/-- Taking all nonzero residual directions gives precisely the operator norm. -/
theorem worst_gain_eq_norm (G : E →L[𝕜] F) :
    sSup ((fun b : E ↦ ‖G b‖ / ‖b‖) '' {b : E | b ≠ 0}) = ‖G‖ := by
  rw [gain_ratio_image_eq_unit_image]
  exact worst_unit_gain_eq_norm G

/-- The variational part of Theorem 5: worst inverse gain is reciprocal
least forward gain, with the latter defined on the unit sphere. -/
theorem amplification_variational [Nontrivial E] (A : E ≃L[𝕜] F) :
    sSup ((fun b : F ↦ ‖A.symm b‖ / ‖b‖) '' {b : F | b ≠ 0}) =
      ‖(A.symm : F →L[𝕜] E)‖ ∧
    ‖(A.symm : F →L[𝕜] E)‖ =
      1 / sInf ((fun v : E ↦ ‖A v‖) '' Metric.sphere 0 1) := by
  refine ⟨worst_gain_eq_norm (A.symm : F →L[𝕜] E), ?_⟩
  change ‖(A.symm : F →L[𝕜] E)‖ =
    1 / sInf ((fun v : E ↦ ‖(A : E →L[𝕜] F) v‖) '' Metric.sphere 0 1)
  rw [← minimumGain_eq_unitSphere (A : E →L[𝕜] F)]
  exact norm_symm_eq_inv_minimumGain A

/-- For any invertible continuous linear map, the inverse-norm formula
uses the independent unit-sphere minimum of the original map. -/
theorem norm_inverse_eq_inv_unit_minimum [Nontrivial E]
    (T : E →L[𝕜] E) (hu : IsUnit T) :
    ‖Ring.inverse T‖ =
      1 / sInf ((fun v : E ↦ ‖T v‖) '' Metric.sphere 0 1) := by
  let e : E ≃L[𝕜] E := ContinuousLinearEquiv.ofUnit hu.unit
  have he : (e : E →L[𝕜] E) = T := hu.unit_spec
  have hinv : (e.symm : E →L[𝕜] E) = Ring.inverse T := by
    change (↑hu.unit⁻¹ : E →L[𝕜] E) = Ring.inverse T
    rw [← Ring.inverse_unit, hu.unit_spec]
  have h := (amplification_variational e).2
  change ‖(e.symm : E →L[𝕜] E)‖ =
    1 / sInf ((fun v : E ↦ ‖(e : E →L[𝕜] E) v‖) '' Metric.sphere 0 1) at h
  simpa only [he, hinv] using h

end MinimumGain

/-- A ring homomorphism transports the inverse of a known unit. -/
lemma map_ring_inverse_of_isUnit {R S : Type*} [Ring R] [Ring S]
    (f : R →+* S) (x : R) (hu : IsUnit x) :
    f (Ring.inverse x) = Ring.inverse (f x) := by
  obtain ⟨u, rfl⟩ := hu
  rw [Ring.inverse_unit]
  change f (↑u⁻¹) = Ring.inverse (↑(Units.map f.toMonoidHom u))
  rw [Ring.inverse_unit]
  rfl

section Spectrum
open scoped ContinuousFunctionalCalculus
variable {A : Type*} [CStarAlgebra A] [Nontrivial A]

/-- Distance from the evaluation point +1 to the complex spectrum. -/
def spectralDistance (K : A) : ℝ :=
  sInf ((fun z : ℂ ↦ ‖1 - z‖) '' spectrum ℂ K)

/-- Every spectral mode gives a lower bound on the inverse norm, whether
or not the Jacobian is normal. -/
theorem reciprocal_spectral_distance_le (K : A) (h1 : (1 : ℂ) ∉ spectrum ℂ K)
    (z : ℂ) (hz : z ∈ spectrum ℂ K) :
    1 / ‖1 - z‖ ≤ ‖Ring.inverse (1 - K)‖ := by
  have hu : IsUnit (1 - K) := by
    simpa using (spectrum.notMem_iff.mp h1)
  have hzA : 1 - z ∈ spectrum ℂ (1 - K) := by
    have hid : algebraMap ℂ A (1 - z) - (1 - K) =
        -(algebraMap ℂ A z - K) := by
      rw [map_sub, map_one]
      noncomm_ring
    rw [spectrum.mem_iff, hid, IsUnit.neg_iff]
    exact spectrum.mem_iff.mp hz
  have hzU : 1 - z ∈ spectrum ℂ (hu.unit : A) := by
    rw [hu.unit_spec]
    exact hzA
  have hzInv : (1 - z)⁻¹ ∈ spectrum ℂ (↑hu.unit⁻¹ : A) := by
    apply (spectrum.inv₀_mem_iff (r := (1 - z)⁻¹) (a := hu.unit)).mp
    simpa using hzU
  have hnorm := spectrum.norm_le_norm_of_mem hzInv
  rw [← Ring.inverse_unit, hu.unit_spec] at hnorm
  simpa [norm_inv, one_div] using hnorm

/-- The nearest spectral point supplies the general lower bound. -/
theorem inverse_spectralDistance_le (K : A) (h1 : (1 : ℂ) ∉ spectrum ℂ K) :
    1 / spectralDistance K ≤ ‖Ring.inverse (1 - K)‖ := by
  have hc : IsCompact ((fun z : ℂ ↦ ‖1 - z‖) '' spectrum ℂ K) :=
    (spectrum.isCompact K).image (continuous_const.sub continuous_id).norm
  obtain ⟨d, hd⟩ := hc.exists_isLeast ((spectrum.nonempty K).image _)
  obtain ⟨z, hz, rfl⟩ := hd.1
  rw [show spectralDistance K = ‖1 - z‖ from hd.csInf_eq]
  exact reciprocal_spectral_distance_le K h1 z hz

/-- For a normal Jacobian, the inverse norm is the greatest reciprocal
spectral distance; this is the normal-matrix spectral theorem. -/
theorem normal_resolvent_spectral_max (K : A) (hn : IsStarNormal K)
    (h1 : (1 : ℂ) ∉ spectrum ℂ K) :
    IsGreatest ((fun z : ℂ ↦ 1 / ‖1 - z‖) '' spectrum ℂ K)
      ‖Ring.inverse (1 - K)‖ := by
  let f : ℂ → ℂ := fun z ↦ 1 - z
  have hf : ContinuousOn f (spectrum ℂ K) :=
    (continuous_const.sub continuous_id).continuousOn
  have hne : ∀ z ∈ spectrum ℂ K, f z ≠ 0 := by
    intro z hz heq
    have heq' : (1 : ℂ) = z := sub_eq_zero.mp heq
    exact h1 (heq'.symm ▸ hz)
  have hfinv : ContinuousOn (fun z ↦ (f z)⁻¹) (spectrum ℂ K) := hf.inv₀ hne
  have hcfc : cfc f K = 1 - K := by
    change cfc (fun z : ℂ ↦ 1 - z) K = 1 - K
    rw [cfc_sub (fun _ : ℂ ↦ 1) (fun z : ℂ ↦ z) K,
      cfc_const_one ℂ K hn, cfc_id' ℂ K hn]
  have hident : cfc (fun z ↦ (f z)⁻¹) K = Ring.inverse (1 - K) := by
    rw [cfc_inv f K hne hf hn, hcfc]
  have hmax := IsGreatest.norm_cfc (fun z ↦ (f z)⁻¹) K hfinv hn
  simpa only [hident, f, norm_inv, one_div] using hmax

/-- Normality upgrades the spectral lower bound to an exact reciprocal
minimum-distance formula. -/
theorem normal_resolvent_norm (K : A) (hn : IsStarNormal K)
    (h1 : (1 : ℂ) ∉ spectrum ℂ K) :
    ‖Ring.inverse (1 - K)‖ = 1 / spectralDistance K := by
  have hmax := normal_resolvent_spectral_max K hn h1
  obtain ⟨z, hz, hval⟩ := hmax.1
  change 1 / ‖1 - z‖ = ‖Ring.inverse (1 - K)‖ at hval
  have hleast : IsLeast ((fun w : ℂ ↦ ‖1 - w‖) '' spectrum ℂ K) ‖1 - z‖ := by
    refine ⟨⟨z, hz, rfl⟩, ?_⟩
    rintro d ⟨w, hw, rfl⟩
    have hp : 0 < ‖1 - w‖ := by
      apply norm_pos_iff.mpr
      intro heq
      exact h1 ((sub_eq_zero.mp heq).symm ▸ hw)
    apply le_of_one_div_le_one_div hp
    rw [hval]
    exact hmax.2 ⟨w, hw, rfl⟩
  rw [show spectralDistance K = ‖1 - z‖ from hleast.csInf_eq]
  exact hval.symm

/-- A positive real spectral mode at the radius gives the scalar gain.
The spectral bound and attained positive mode are stated explicitly. -/
theorem normal_positive_mode_gain (K : A) (hn : IsStarNormal K)
    (f : ℝ) (_hf0 : 0 ≤ f) (hf : f < 1)
    (hfmem : (f : ℂ) ∈ spectrum ℂ K)
    (hbound : ∀ z ∈ spectrum ℂ K, ‖z‖ ≤ f) :
    ‖Ring.inverse (1 - K)‖ = 1 / (1 - f) := by
  have h1 : (1 : ℂ) ∉ spectrum ℂ K := by
    intro h
    have h' := hbound 1 h
    simp only [norm_one] at h'
    exact (not_le_of_gt hf) h'
  have hmax := normal_resolvent_spectral_max K hn h1
  have hdist : ‖1 - (f : ℂ)‖ = 1 - f := by
    rw [← Complex.ofReal_one, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (sub_nonneg.mpr hf.le)]
  apply le_antisymm
  · obtain ⟨z, hz, hval⟩ := hmax.1
    rw [← hval]
    apply one_div_le_one_div_of_le (sub_pos.mpr hf)
    calc
      1 - f ≤ 1 - ‖z‖ := sub_le_sub_left (hbound z hz) 1
      _ ≤ ‖1 - z‖ := by simpa using norm_sub_norm_le (1 : ℂ) z
  · have h := hmax.2 ⟨(f : ℂ), hfmem, rfl⟩
    simpa [hdist] using h

/-- The scalar formula applies precisely when the nonnegative real
spectral-radius value itself is a spectral mode of a normal Jacobian. -/
theorem normal_positive_radius_gain (K : A) (hn : IsStarNormal K)
    (f : ℝ≥0) (hf : (f : ℝ) < 1)
    (hradius : spectralRadius ℂ K = (f : ℝ≥0∞))
    (hfmem : (f : ℂ) ∈ spectrum ℂ K) :
    ‖Ring.inverse (1 - K)‖ = 1 / (1 - (f : ℝ)) := by
  apply normal_positive_mode_gain K hn (f : ℝ) f.coe_nonneg hf hfmem
  intro z hz
  have hle : (‖z‖₊ : ℝ≥0∞) ≤ spectralRadius ℂ K := by
    rw [spectralRadius_eq_of_unital]
    exact le_iSup_of_le z (le_iSup_of_le hz le_rfl)
  rw [hradius] at hle
  exact_mod_cast (ENNReal.coe_le_coe.mp hle)

end Spectrum
section RealComplexNorm
open scoped Matrix.Norms.L2Operator
variable {n : Type*} [Fintype n] [DecidableEq n]

/-- In finite dimensions the algebraic spectrum used below is exactly
the set of eigenvalues of the matrix acting on coordinate vectors. -/
theorem matrix_spectrum_eigenvalue_bridge (K : Matrix n n ℂ) (z : ℂ) :
    z ∈ spectrum ℂ K ↔ Module.End.HasEigenvalue K.toLin' z := by
  rw [← Matrix.spectrum_toLin']
  exact Module.End.hasEigenvalue_iff_mem_spectrum.symm

private def realVec (z : EuclideanSpace ℂ n) : EuclideanSpace ℝ n :=
  WithLp.toLp 2 (fun i ↦ (z i).re)
private def imagVec (z : EuclideanSpace ℂ n) : EuclideanSpace ℝ n :=
  WithLp.toLp 2 (fun i ↦ (z i).im)
private def complexVec (x : EuclideanSpace ℝ n) : EuclideanSpace ℂ n :=
  WithLp.toLp 2 (fun i ↦ (x i : ℂ))

omit [DecidableEq n] in
private lemma norm_sq_real_imag (z : EuclideanSpace ℂ n) :
    ‖z‖ ^ 2 = ‖realVec z‖ ^ 2 + ‖imagVec z‖ ^ 2 := by
  simp only [EuclideanSpace.norm_sq_eq, realVec, imagVec, Real.norm_eq_abs, sq_abs]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  simpa only [Complex.normSq_apply, sq] using Complex.sq_norm (z i)

omit [DecidableEq n] in
private lemma norm_complexVec (x : EuclideanSpace ℝ n) : ‖complexVec x‖ = ‖x‖ := by
  apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  simp [EuclideanSpace.norm_sq_eq, complexVec]

private lemma realVec_matrix (K : Matrix n n ℝ) (z : EuclideanSpace ℂ n) :
    realVec ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := n)) (K.map (algebraMap ℝ ℂ)) z) =
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K (realVec z) := by
  apply PiLp.ext
  intro i
  simp [realVec, Matrix.ofLp_toEuclideanCLM, Matrix.mulVec, dotProduct, Complex.mul_re]

private lemma imagVec_matrix (K : Matrix n n ℝ) (z : EuclideanSpace ℂ n) :
    imagVec ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := n)) (K.map (algebraMap ℝ ℂ)) z) =
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K (imagVec z) := by
  apply PiLp.ext
  intro i
  simp [imagVec, Matrix.ofLp_toEuclideanCLM, Matrix.mulVec, dotProduct, Complex.mul_im]

private lemma complexVec_matrix (K : Matrix n n ℝ) (x : EuclideanSpace ℝ n) :
    (Matrix.toEuclideanCLM (𝕜 := ℂ) (n := n)) (K.map (algebraMap ℝ ℂ)) (complexVec x) =
      complexVec ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K x) := by
  apply PiLp.ext
  intro i
  simp [complexVec, Matrix.ofLp_toEuclideanCLM, Matrix.mulVec, dotProduct]

/-- Complexifying a real matrix preserves its Euclidean operator norm.
This bridges the real-state gain to the complex-spectrum formulas. -/
theorem real_matrix_complexification_norm (K : Matrix n n ℝ) :
    ‖K.map (algebraMap ℝ ℂ)‖ = ‖K‖ := by
  rw [← Matrix.l2_opNorm_toEuclideanCLM, ← Matrix.l2_opNorm_toEuclideanCLM]
  let T := (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K
  let C := (Matrix.toEuclideanCLM (𝕜 := ℂ) (n := n)) (K.map (algebraMap ℝ ℂ))
  change ‖C‖ = ‖T‖
  apply le_antisymm
  · apply C.opNorm_le_bound (norm_nonneg T)
    intro z
    apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (norm_nonneg _) (norm_nonneg _))).mp
    have hr := T.le_opNorm (realVec z)
    have hi := T.le_opNorm (imagVec z)
    have hr2 := mul_self_le_mul_self (norm_nonneg _) hr
    have hi2 := mul_self_le_mul_self (norm_nonneg _) hi
    have hz := norm_sq_real_imag z
    have hC := norm_sq_real_imag (C z)
    rw [realVec_matrix, imagVec_matrix] at hC
    change ‖C z‖ ^ 2 = ‖T (realVec z)‖ ^ 2 + ‖T (imagVec z)‖ ^ 2 at hC
    have hzT := congrArg (fun a : ℝ ↦ ‖T‖ ^ 2 * a) hz
    nlinarith only [hr2, hi2, hC, hzT]
  · apply T.opNorm_le_bound (norm_nonneg C)
    intro x
    have h := C.le_opNorm (complexVec x)
    change ‖(Matrix.toEuclideanCLM (𝕜 := ℂ) (n := n)) (K.map (algebraMap ℝ ℂ)) (complexVec x)‖ ≤
      ‖C‖ * ‖complexVec x‖ at h
    rw [complexVec_matrix, norm_complexVec, norm_complexVec] at h
    exact h

/-- Complexification also preserves normality. -/
theorem real_matrix_complexification_normal (K : Matrix n n ℝ)
    (hn : IsStarNormal K) : IsStarNormal (K.map (algebraMap ℝ ℂ)) := by
  let f : Matrix n n ℝ →+* Matrix n n ℂ := (algebraMap ℝ ℂ).mapMatrix
  have hs : f (star K) = star (f K) := by
    ext i j
    simp [f, Matrix.star_apply]
  refine ⟨?_⟩
  change star (f K) * f K = f K * star (f K)
  have h := congrArg f hn.star_comm_self.eq
  simpa only [map_mul, hs] using h

/-- The real inverse gain is exactly the gain of its complexification. -/
theorem real_matrix_resolvent_norm_complexification (K : Matrix n n ℝ)
    (hu : IsUnit (1 - K)) :
    ‖Ring.inverse (1 - K.map (algebraMap ℝ ℂ))‖ = ‖Ring.inverse (1 - K)‖ := by
  let f : Matrix n n ℝ →+* Matrix n n ℂ := (algebraMap ℝ ℂ).mapMatrix
  have hi := map_ring_inverse_of_isUnit f (1 - K) hu
  simp only [map_sub, map_one] at hi
  change ‖Ring.inverse (1 - f K)‖ = ‖Ring.inverse (1 - K)‖
  rw [← hi]
  exact real_matrix_complexification_norm _

/-- Theorem 5 for real Euclidean matrices, including all complex-spectrum
claims transferred through a proved equality of Euclidean operator norms. -/
theorem real_matrix_resolvent_amplification [Nonempty n]
    (K : Matrix n n ℝ) (hu : IsUnit (1 - K)) :
    sSup ((fun b : EuclideanSpace ℝ n ↦
      ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b‖ / ‖b‖) '' {b | b ≠ 0}) =
      ‖Ring.inverse (1 - K)‖ ∧
    ‖Ring.inverse (1 - K)‖ =
      1 / sInf ((fun v : EuclideanSpace ℝ n ↦
        ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (1 - K)) v‖) '' Metric.sphere 0 1) ∧
    1 / spectralDistance (K.map (algebraMap ℝ ℂ)) ≤ ‖Ring.inverse (1 - K)‖ ∧
    (IsStarNormal K → ‖Ring.inverse (1 - K)‖ =
      1 / spectralDistance (K.map (algebraMap ℝ ℂ))) ∧
    (∀ (f : ℝ≥0), (f : ℝ) < 1 →
      spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) = (f : ℝ≥0∞) →
      (f : ℂ) ∈ spectrum ℂ (K.map (algebraMap ℝ ℂ)) → IsStarNormal K →
      ‖Ring.inverse (1 - K)‖ = 1 / (1 - (f : ℝ))) := by
  let Kc := K.map (algebraMap ℝ ℂ)
  let fm : Matrix n n ℝ →+* Matrix n n ℂ := (algebraMap ℝ ℂ).mapMatrix
  have huc : IsUnit (1 - Kc) := by
    change IsUnit (1 - fm K)
    simpa only [map_sub, map_one] using hu.map fm
  have h1 : (1 : ℂ) ∉ spectrum ℂ Kc := by
    apply spectrum.notMem_iff.mpr
    simpa using huc
  have hnorm := real_matrix_resolvent_norm_complexification K hu
  have ht : IsUnit ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (1 - K)) :=
    hu.map (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)).toRingHom
  have hti : (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K)) =
      Ring.inverse ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (1 - K)) :=
    map_ring_inverse_of_isUnit (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)).toRingHom (1 - K) hu
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa only [Matrix.l2_opNorm_toEuclideanCLM] using
      worst_gain_eq_norm ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K)))
  · rw [← Matrix.l2_opNorm_toEuclideanCLM, hti]
    exact norm_inverse_eq_inv_unit_minimum _ ht
  · rw [← hnorm]
    exact inverse_spectralDistance_le Kc h1
  · intro hn
    rw [← hnorm]
    exact normal_resolvent_norm Kc (real_matrix_complexification_normal K hn) h1
  · intro f hf hr hm hn
    rw [← hnorm]
    exact normal_positive_radius_gain Kc (real_matrix_complexification_normal K hn)
      f hf hr hm

end RealComplexNorm

/-- Theorem 5, collected for complex Euclidean or Hilbert state spaces:
worst directional gain, least forward gain, the general spectral lower
bound, the normal equality, and the positive-radius scalar corollary. -/
theorem resolvent_amplification {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℂ E] [CompleteSpace E] [Nontrivial E]
    (K : E →L[ℂ] E) (h1 : (1 : ℂ) ∉ spectrum ℂ K) :
    sSup ((fun b : E ↦ ‖(Ring.inverse (1 - K)) b‖ / ‖b‖) '' {b : E | b ≠ 0}) =
      ‖Ring.inverse (1 - K)‖ ∧
    ‖Ring.inverse (1 - K)‖ =
      1 / sInf ((fun v : E ↦ ‖(1 - K) v‖) '' Metric.sphere 0 1) ∧
    1 / spectralDistance K ≤ ‖Ring.inverse (1 - K)‖ ∧
    (IsStarNormal K → ‖Ring.inverse (1 - K)‖ = 1 / spectralDistance K) ∧
    (∀ (f : ℝ≥0), (f : ℝ) < 1 → spectralRadius ℂ K = (f : ℝ≥0∞) →
      (f : ℂ) ∈ spectrum ℂ K → IsStarNormal K →
      ‖Ring.inverse (1 - K)‖ = 1 / (1 - (f : ℝ))) := by
  refine ⟨worst_gain_eq_norm _, ?_, inverse_spectralDistance_le K h1,
    (fun hn ↦ normal_resolvent_norm K hn h1), ?_⟩
  · apply norm_inverse_eq_inv_unit_minimum
    simpa using (spectrum.notMem_iff.mp h1)
  · intro f hf hr hm hn
    exact normal_positive_radius_gain K hn f hf hr hm

open scoped Matrix.Norms.L2Operator

/-- Theorem 5, complete manuscript-facing declaration: both real and
complex state spaces have the variational inverse gain, the general
spectral lower bound, the normal equality, and the positive-radius formula. -/
theorem worst_case_amplification {E n : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℂ E] [CompleteSpace E] [Nontrivial E]
    [Fintype n] [DecidableEq n] [Nonempty n] :
    (∀ (K : E →L[ℂ] E), (1 : ℂ) ∉ spectrum ℂ K →
      (sSup ((fun b : E ↦ ‖(Ring.inverse (1 - K)) b‖ / ‖b‖) '' {b : E | b ≠ 0}) =
      ‖Ring.inverse (1 - K)‖ ∧
    ‖Ring.inverse (1 - K)‖ =
      1 / sInf ((fun v : E ↦ ‖(1 - K) v‖) '' Metric.sphere 0 1) ∧
    1 / spectralDistance K ≤ ‖Ring.inverse (1 - K)‖ ∧
    (IsStarNormal K → ‖Ring.inverse (1 - K)‖ = 1 / spectralDistance K) ∧
    (∀ (f : ℝ≥0), (f : ℝ) < 1 → spectralRadius ℂ K = (f : ℝ≥0∞) →
      (f : ℂ) ∈ spectrum ℂ K → IsStarNormal K →
      ‖Ring.inverse (1 - K)‖ = 1 / (1 - (f : ℝ))))) ∧
    (∀ (K : Matrix n n ℝ), IsUnit (1 - K) →
      (sSup ((fun b : EuclideanSpace ℝ n ↦
      ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b‖ / ‖b‖) '' {b | b ≠ 0}) =
      ‖Ring.inverse (1 - K)‖ ∧
    ‖Ring.inverse (1 - K)‖ =
      1 / sInf ((fun v : EuclideanSpace ℝ n ↦
        ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (1 - K)) v‖) '' Metric.sphere 0 1) ∧
    1 / spectralDistance (K.map (algebraMap ℝ ℂ)) ≤ ‖Ring.inverse (1 - K)‖ ∧
    (IsStarNormal K → ‖Ring.inverse (1 - K)‖ =
      1 / spectralDistance (K.map (algebraMap ℝ ℂ))) ∧
    (∀ (f : ℝ≥0), (f : ℝ) < 1 →
      spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) = (f : ℝ≥0∞) →
      (f : ℂ) ∈ spectrum ℂ (K.map (algebraMap ℝ ℂ)) → IsStarNormal K →
      ‖Ring.inverse (1 - K)‖ = 1 / (1 - (f : ℝ))))) := by
  exact ⟨fun K h ↦ resolvent_amplification K h,
    fun K h ↦ real_matrix_resolvent_amplification K h⟩

end Grail
