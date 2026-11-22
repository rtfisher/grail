import Mathlib.Analysis.Normed.Algebra.GelfandFormula
import Mathlib.Analysis.Normed.Operator.NormedSpace
import Mathlib.Analysis.Normed.Operator.Bilinear
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Complex.OperatorNorm
import Mathlib.Tactic.NoncommRing

/-!
# GRAIL resolvent identities and repeated linear feedback

Theorems 3 and 4. The spectral-radius result is proved for complex Banach
algebras and transferred explicitly to real Euclidean matrices through
complexification. All continuous-linear-map norms are operator norms;
the real-matrix section uses the Euclidean operator norm.
-/

noncomputable section
open scoped Topology
open Filter

namespace Grail

/-- A resolvent is explicitly identified by its two inverse identities. -/
def IsResolvent {R : Type*} [Ring R] (K G : R) : Prop :=
  G * (1 - K) = 1 ∧ (1 - K) * G = 1

/-- Changing a Jacobian changes the inverse by the difference sandwiched
between the new and old resolvents. No commutativity is assumed. -/
theorem resolvent_difference {R : Type*} [Ring R]
    (Kt Ke Gt Ge : R) (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) :
    Ge - Gt = Ge * (Ke - Kt) * Gt := by
  have hd : Ke - Kt = (1 - Kt) - (1 - Ke) := by noncomm_ring
  rw [hd, mul_sub, sub_mul, mul_assoc, ht.2, mul_one, he.1, one_mul]

/-- The norm cost of Jacobian substitution includes both inverse norms. -/
theorem norm_resolvent_difference_le {R : Type*} [NormedRing R]
    (Kt Ke Gt Ge : R) (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) :
    ‖Ge - Gt‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ := by
  rw [resolvent_difference Kt Ke Gt Ge ht he]
  exact (norm_mul_le _ _).trans
    (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _))

/-- The Neumann sum has both inverse identities whenever it is summable. -/
theorem isResolvent_tsum_pow {R : Type*} [NormedRing R]
    (K : R) (h : Summable (fun n : ℕ ↦ K ^ n)) :
    IsResolvent K (∑' n : ℕ, K ^ n) := by
  exact ⟨h.tsum_pow_mul_one_sub, h.one_sub_mul_tsum_pow⟩

/-- Norm contraction suffices for the Neumann representation. -/
theorem hasSum_resolvent_of_norm_lt_one {R : Type*} [NormedRing R]
    [CompleteSpace R] (K : R) (h : ‖K‖ < 1) :
    HasSum (fun n : ℕ ↦ K ^ n) (Ring.inverse (1 - K)) :=
  hasSum_geom_series_inverse K h

/-- Spectral-radius contraction implies summability, without requiring
that the operator norm itself be below one. -/
theorem summable_pow_of_spectralRadius_lt_one {R : Type*}
    [NormedRing R] [NormedAlgebra ℂ R] [CompleteSpace R]
    (K : R) (h : spectralRadius ℂ K < 1) :
    Summable (fun n : ℕ ↦ K ^ n) := by
  have hg : gelfandRadius K < 1 := by
    rw [spectrum.spectralRadius_eq_gelfandRadius] at h
    exact (ENNReal.ofReal_lt_one).mp h
  obtain ⟨r, hgr, hr⟩ := exists_between hg
  obtain ⟨C, hC, hbound⟩ :=
    norm_pow_le_mul_pow_of_gelfandRadius_lt K hgr
  have hr0 : 0 ≤ r := (gelfandRadius_nonneg K).trans hgr.le
  exact ((summable_geometric_of_abs_lt_one
    (by rwa [abs_of_nonneg hr0])).mul_left C).of_norm_bounded hbound

/-- Theorem 4: the full spectral-radius hypothesis gives the operator
Neumann/Born series, including non-normal cases with norm above one. -/
theorem hasSum_resolvent_of_spectralRadius_lt_one {R : Type*}
    [NormedRing R] [NormedAlgebra ℂ R] [CompleteSpace R]
    (K : R) (h : spectralRadius ℂ K < 1) :
    HasSum (fun n : ℕ ↦ K ^ n) (Ring.inverse (1 - K)) := by
  have hs := summable_pow_of_spectralRadius_lt_one K h
  let u : Rˣ :=
    { val := 1 - K
      inv := ∑' n : ℕ, K ^ n
      val_inv := hs.one_sub_mul_tsum_pow
      inv_val := hs.tsum_pow_mul_one_sub }
  have hinv : Ring.inverse (1 - K) = ∑' n : ℕ, K ^ n := by
    change Ring.inverse (u : R) = _
    rw [Ring.inverse_unit]
    rfl
  rw [hinv]
  exact hs.hasSum

/-- The operator tail of a norm-contractive Neumann series has a geometric bound. -/
theorem resolvent_series_tail_bound {R : Type*} [NormedRing R] [NormOneClass R]
    [CompleteSpace R] (K : R) (h : ‖K‖ < 1) (m : ℕ) :
    ‖Ring.inverse (1 - K) - ∑ j ∈ Finset.range (m + 1), K ^ j‖ ≤
      ‖K‖ ^ (m + 1) / (1 - ‖K‖) := by
  have hb (n : ℕ) : ‖K ^ n‖ ≤ 1 * ‖K‖ ^ n := by
    simpa using norm_pow_le K n
  simpa [norm_sub_rev] using norm_sub_le_of_geometric_bound_of_hasSum h hb
    (hasSum_geom_series_inverse K h) (m + 1)

/-- Theorem 3, collected as one manuscript-facing declaration. -/
theorem jacobian_substitution {R : Type*} [NormedRing R]
    (Kt Ke Gt Ge : R) (ht : IsResolvent Kt Gt) (he : IsResolvent Ke Ge) :
    Ge - Gt = Ge * (Ke - Kt) * Gt ∧
      ‖Ge - Gt‖ ≤ ‖Ge‖ * ‖Ke - Kt‖ * ‖Gt‖ :=
  ⟨resolvent_difference Kt Ke Gt Ge ht he,
    norm_resolvent_difference_le Kt Ke Gt Ge ht he⟩

section LinearResponse
variable {𝕜 E : Type*} [NontriviallyNormedField 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E]

/-- The exact fixed-point response equation has a unique solution once
its inverse is supplied. This statement concerns a linearized cycle. -/
theorem linear_response_iff (K G : E →L[𝕜] E)
    (hG : IsResolvent K G) (e b : E) :
    e = K e + b ↔ e = G b := by
  constructor
  · intro he
    have hAe : (1 - K) e = b := by
      simp only [sub_apply, one_apply_eq_self]
      exact sub_eq_iff_eq_add.mpr (by simpa [add_comm] using he)
    have h := congrArg (fun A : E →L[𝕜] E ↦ A e) hG.1
    simp only [mul_apply_eq_comp, one_apply_eq_self] at h
    rw [← h, hAe]
  · intro he
    subst e
    have h := congrArg (fun A : E →L[𝕜] E ↦ A b) hG.2
    simp only [mul_apply_eq_comp, sub_apply,
      one_apply_eq_self] at h
    exact sub_eq_iff_eq_add.mp h |>.trans (add_comm _ _)

end LinearResponse

/-- The Neumann series may be applied term by term to any residual. -/
theorem neumann_action_series {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℂ E] [CompleteSpace E]
    (K : E →L[ℂ] E) (h : spectralRadius ℂ K < 1) (b : E) :
    HasSum (fun j : ℕ ↦ (K ^ j) b) ((Ring.inverse (1 - K)) b) := by
  exact (hasSum_resolvent_of_spectralRadius_lt_one K h).mapL
    (ContinuousLinearMap.apply ℂ E b)

/-- Theorem 4, collected: spectral contraction sums all linear feedback
and gives the unique solution of the linear response equation. -/
theorem neumann_linear_response {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℂ E] [CompleteSpace E]
    (K : E →L[ℂ] E) (h : spectralRadius ℂ K < 1) :
    HasSum (fun n : ℕ ↦ K ^ n) (Ring.inverse (1 - K)) ∧
      ∀ e b : E, e = K e + b ↔ e = (Ring.inverse (1 - K)) b := by
  have hs := hasSum_resolvent_of_spectralRadius_lt_one K h
  refine ⟨hs, ?_⟩
  have hg : IsResolvent K (Ring.inverse (1 - K)) := by
    rw [← hs.tsum_eq]
    exact isResolvent_tsum_pow K hs.summable
  exact linear_response_iff K (Ring.inverse (1 - K)) hg

/-- Applying the operator tail bound to a physical residual vector gives
its geometric response-error estimate in the same vector norm. -/
theorem resolvent_action_tail_bound {𝕜 E : Type*} [NontriviallyNormedField 𝕜]
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [CompleteSpace E] [Nontrivial E]
    (K : E →L[𝕜] E) (h : ‖K‖ < 1) (m : ℕ) (b : E) :
    ‖(Ring.inverse (1 - K)) b -
        (∑ j ∈ Finset.range (m + 1), K ^ j) b‖ ≤
      ‖K‖ ^ (m + 1) / (1 - ‖K‖) * ‖b‖ := by
  calc
    _ = ‖(Ring.inverse (1 - K) - ∑ j ∈ Finset.range (m + 1), K ^ j) b‖ := by
      rw [sub_apply]
    _ ≤ ‖Ring.inverse (1 - K) - ∑ j ∈ Finset.range (m + 1), K ^ j‖ * ‖b‖ :=
      ContinuousLinearMap.le_opNorm _ b
    _ ≤ _ := mul_le_mul_of_nonneg_right
      (resolvent_series_tail_bound K h m) (norm_nonneg b)

/-- The complete complex-state statement of Theorem 4: norm-convergent
operator and response series, unique fixed point, and the conditional tail bound. -/
theorem neumann_response_full {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℂ E] [CompleteSpace E] [Nontrivial E]
    (K : E →L[ℂ] E) (h : spectralRadius ℂ K < 1) :
    HasSum (fun j : ℕ ↦ K ^ j) (Ring.inverse (1 - K)) ∧
    (∀ b : E, HasSum (fun j : ℕ ↦ (K ^ j) b) ((Ring.inverse (1 - K)) b)) ∧
    (∀ e b : E, e = K e + b ↔ e = (Ring.inverse (1 - K)) b) ∧
    (‖K‖ < 1 → ∀ (m : ℕ) (b : E),
      ‖(Ring.inverse (1 - K)) b - ∑ j ∈ Finset.range (m + 1), (K ^ j) b‖ ≤
        ‖K‖ ^ (m + 1) / (1 - ‖K‖) * ‖b‖) := by
  refine ⟨hasSum_resolvent_of_spectralRadius_lt_one K h,
    neumann_action_series K h, (neumann_linear_response K h).2, ?_⟩
  intro hn m b
  simpa only [sum_apply] using resolvent_action_tail_bound K hn m b

section RealMatrices
open scoped Matrix.Norms.L2Operator
variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Real matrices use their complex spectrum: convergence transfers back
through entrywise real part, which is continuous in finite dimension.
The matrix norm in this section is the Euclidean operator norm. -/
theorem summable_real_matrix_powers (K : Matrix n n ℝ)
    (h : spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) < 1) :
    Summable (fun j : ℕ ↦ K ^ j) := by
  let reM : Matrix n n ℂ →L[ℝ] Matrix n n ℝ :=
    (Complex.reCLM.toLinearMap.mapMatrix).toContinuousLinearMap
  have hre (j : ℕ) : reM ((K.map (algebraMap ℝ ℂ)) ^ j) = K ^ j := by
    rw [← Matrix.map_pow]
    ext i k
    simp [reM]
  have hs := summable_pow_of_spectralRadius_lt_one (K.map (algebraMap ℝ ℂ)) h
  simpa only [hre] using (hs.mapL reM)

/-- The full spectral-radius result for real Jacobian matrices, without
assuming the stronger Euclidean operator-norm contraction condition. -/
theorem real_matrix_neumann_series (K : Matrix n n ℝ)
    (h : spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) < 1) :
    HasSum (fun j : ℕ ↦ K ^ j) (Ring.inverse (1 - K)) := by
  have hs := summable_real_matrix_powers K h
  let u : (Matrix n n ℝ)ˣ :=
    { val := 1 - K
      inv := ∑' j : ℕ, K ^ j
      val_inv := hs.one_sub_mul_tsum_pow
      inv_val := hs.tsum_pow_mul_one_sub }
  have hinv : Ring.inverse (1 - K) = ∑' j : ℕ, K ^ j := by
    change Ring.inverse (u : Matrix n n ℝ) = _
    rw [Ring.inverse_unit]
    rfl
  rw [hinv]
  exact hs.hasSum

/-- Theorem 4 for real Euclidean states: the complex spectral-radius
condition sums the real matrix series and solves the real feedback equation. -/
theorem real_matrix_neumann_response (K : Matrix n n ℝ)
    (h : spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) < 1) :
    HasSum (fun j : ℕ ↦ K ^ j) (Ring.inverse (1 - K)) ∧
    ∀ e b : EuclideanSpace ℝ n,
      e = ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K) e + b ↔
        e = ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b := by
  have hs := real_matrix_neumann_series K h
  refine ⟨hs, ?_⟩
  have hg : IsResolvent K (Ring.inverse (1 - K)) := by
    rw [← hs.tsum_eq]
    exact isResolvent_tsum_pow K hs.summable
  have hmap : IsResolvent ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K)
      ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) := by
    constructor
    · have hm := congrArg (fun M : Matrix n n ℝ ↦ (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) M) hg.1
      simpa only [map_mul, map_sub, map_one] using hm
    · have hm := congrArg (fun M : Matrix n n ℝ ↦ (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) M) hg.2
      simpa only [map_mul, map_sub, map_one] using hm
  exact linear_response_iff _ _ hmap

/-- Real physical residuals also admit the term-by-term response series. -/
theorem real_matrix_neumann_action (K : Matrix n n ℝ)
    (h : spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) < 1)
    (b : EuclideanSpace ℝ n) :
    HasSum (fun j : ℕ ↦ ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (K ^ j)) b)
      (((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b) := by
  let L : Matrix n n ℝ →L[ℝ] (EuclideanSpace ℝ n →L[ℝ] EuclideanSpace ℝ n) :=
    (Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)).toAlgEquiv.toLinearEquiv.toLinearMap.toContinuousLinearMap
  exact (real_matrix_neumann_series K h).mapL
    ((ContinuousLinearMap.apply ℝ (EuclideanSpace ℝ n) b).comp L)

/-- The real Euclidean tail is bounded in the physical state norm. -/
theorem real_matrix_neumann_tail [Nonempty n] (K : Matrix n n ℝ)
    (h : ‖K‖ < 1) (m : ℕ) (b : EuclideanSpace ℝ n) :
    ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b -
      ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n))
        (∑ j ∈ Finset.range (m + 1), K ^ j)) b‖ ≤
      ‖K‖ ^ (m + 1) / (1 - ‖K‖) * ‖b‖ := by
  let f := Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)
  calc
    _ = ‖(f (Ring.inverse (1 - K) - ∑ j ∈ Finset.range (m + 1), K ^ j)) b‖ := by
      rw [map_sub, sub_apply]
    _ ≤ ‖f (Ring.inverse (1 - K) - ∑ j ∈ Finset.range (m + 1), K ^ j)‖ * ‖b‖ :=
      ContinuousLinearMap.le_opNorm _ b
    _ = ‖Ring.inverse (1 - K) - ∑ j ∈ Finset.range (m + 1), K ^ j‖ * ‖b‖ := rfl
    _ ≤ _ := mul_le_mul_of_nonneg_right
      (resolvent_series_tail_bound K h m) (norm_nonneg b)

/-- The complete real Euclidean-matrix statement of Theorem 4. The
radius is computed from the proved complexification representation. -/
theorem real_matrix_neumann_full [Nonempty n] (K : Matrix n n ℝ)
    (h : spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) < 1) :
    HasSum (fun j : ℕ ↦ K ^ j) (Ring.inverse (1 - K)) ∧
    (∀ b : EuclideanSpace ℝ n,
      HasSum (fun j : ℕ ↦ ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (K ^ j)) b)
        (((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b)) ∧
    (∀ e b : EuclideanSpace ℝ n,
      e = ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K) e + b ↔
        e = ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b) ∧
    (‖K‖ < 1 → ∀ (m : ℕ) (b : EuclideanSpace ℝ n),
      ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b -
        ∑ j ∈ Finset.range (m + 1), ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (K ^ j)) b‖ ≤
        ‖K‖ ^ (m + 1) / (1 - ‖K‖) * ‖b‖) := by
  refine ⟨real_matrix_neumann_series K h, real_matrix_neumann_action K h,
    (real_matrix_neumann_response K h).2, ?_⟩
  intro hn m b
  simpa only [map_sum, sum_apply] using real_matrix_neumann_tail K hn m b

end RealMatrices

open scoped Matrix.Norms.L2Operator

/-- Theorem 4, complete manuscript-facing declaration for complex states
and real Euclidean matrices. Both cases include the action series,
uniqueness, and the norm-contractive geometric tail estimate. -/
theorem neumann_feedback {E n : Type*} [NormedAddCommGroup E]
    [NormedSpace ℂ E] [CompleteSpace E] [Nontrivial E]
    [Fintype n] [DecidableEq n] [Nonempty n] :
    (∀ (K : E →L[ℂ] E), spectralRadius ℂ K < 1 →
      (HasSum (fun j : ℕ ↦ K ^ j) (Ring.inverse (1 - K)) ∧
    (∀ b : E, HasSum (fun j : ℕ ↦ (K ^ j) b) ((Ring.inverse (1 - K)) b)) ∧
    (∀ e b : E, e = K e + b ↔ e = (Ring.inverse (1 - K)) b) ∧
    (‖K‖ < 1 → ∀ (m : ℕ) (b : E),
      ‖(Ring.inverse (1 - K)) b - ∑ j ∈ Finset.range (m + 1), (K ^ j) b‖ ≤
        ‖K‖ ^ (m + 1) / (1 - ‖K‖) * ‖b‖))) ∧
    (∀ (K : Matrix n n ℝ), spectralRadius ℂ (K.map (algebraMap ℝ ℂ)) < 1 →
      (HasSum (fun j : ℕ ↦ K ^ j) (Ring.inverse (1 - K)) ∧
    (∀ b : EuclideanSpace ℝ n,
      HasSum (fun j : ℕ ↦ ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (K ^ j)) b)
        (((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b)) ∧
    (∀ e b : EuclideanSpace ℝ n,
      e = ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) K) e + b ↔
        e = ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b) ∧
    (‖K‖ < 1 → ∀ (m : ℕ) (b : EuclideanSpace ℝ n),
      ‖((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (Ring.inverse (1 - K))) b -
        ∑ j ∈ Finset.range (m + 1), ((Matrix.toEuclideanCLM (𝕜 := ℝ) (n := n)) (K ^ j)) b‖ ≤
        ‖K‖ ^ (m + 1) / (1 - ‖K‖) * ‖b‖))) := by
  exact ⟨fun K h ↦ neumann_response_full K h,
    fun K h ↦ real_matrix_neumann_full K h⟩

end Grail
