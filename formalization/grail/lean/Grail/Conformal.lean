/-
Copyright (c) 2026 GRAIL contributors.
Released under Apache 2.0 license as described in the repository LICENSE.

Finite-sample split-conformal coverage from full finite exchangeability.
The proof allows arbitrary ties and does not assume uniform ranks.
-/
import Grail.CalibrationQuantile
import Mathlib.Data.Finset.Max
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.MeasurableSpace.Embedding
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum

open MeasureTheory Finset
namespace Grail.Conformal

noncomputable def strictRank {ι : Type*} [Fintype ι] (s : ι → ℝ) (i : ι) : ℕ := by
  classical
  exact (Finset.univ.filter fun j => s j < s i).card

theorem card_covered_ge {ι : Type*} [Fintype ι] (s : ι → ℝ) (k : ℕ)
    (hk : k ≤ Fintype.card ι) :
    k ≤ (Finset.univ.filter fun i => strictRank s i < k).card := by
  classical
  let bad := Finset.univ.filter fun i => k ≤ strictRank s i
  by_cases hbad : bad.Nonempty
  · obtain ⟨i, hi, hmin⟩ := bad.exists_min_image s hbad
    have hiRank : k ≤ strictRank s i := (Finset.mem_filter.mp hi).2
    have hsub : (Finset.univ.filter fun j => s j < s i) ⊆
        (Finset.univ.filter fun j => strictRank s j < k) := by
      intro j hj
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
      by_contra hnot
      have hjbad : j ∈ bad := by simp [bad, Nat.le_of_not_gt hnot]
      exact (not_lt_of_ge (hmin j hjbad)) hj
    exact hiRank.trans (Finset.card_le_card hsub)
  · have hgood : ∀ i : ι, strictRank s i < k := by
      intro i
      by_contra hi
      apply hbad
      exact ⟨i, by simp [bad, Nat.le_of_not_gt hi]⟩
    simpa [hgood] using hk

/-- Reindex a finite score vector by a coordinate permutation. -/
noncomputable def reindex {ι : Type*} (e : Equiv.Perm ι) : (ι → ℝ) ≃ᵐ (ι → ℝ) :=
  MeasurableEquiv.piCongrLeft (fun _ : ι => ℝ) e.symm

@[simp] theorem reindex_apply {ι : Type*} (e : Equiv.Perm ι) (s : ι → ℝ) (i : ι) :
    reindex e s i = s (e i) := by
  simp [reindex, MeasurableEquiv.coe_piCongrLeft, Equiv.piCongrLeft_apply]

theorem strictRank_reindex {ι : Type*} [Fintype ι] (e : Equiv.Perm ι)
    (s : ι → ℝ) (i : ι) : strictRank (reindex e s) i = strictRank s (e i) := by
  classical
  unfold strictRank
  apply Finset.card_bij (fun j _ => e j)
  · intro j hj
    simpa using hj
  · intro a ha b hb hab
    exact e.injective hab
  · intro j hj
    refine ⟨e.symm j, ?_, by simp⟩
    simpa using hj

/-- Finite exchangeability is invariance of the full joint law, not just equality of marginals. -/
def ExchangeableLaw {ι : Type*} (μ : Measure (ι → ℝ)) : Prop :=
  ∀ e : Equiv.Perm ι, MeasurePreserving (reindex e) μ μ

noncomputable def coveredEvent {ι : Type*} [Fintype ι] (k : ℕ) (i : ι) : Set (ι → ℝ) :=
  {s | strictRank s i < k}

noncomputable def coverInd {ι : Type*} [Fintype ι] (k : ℕ) (i : ι) : (ι → ℝ) → ℝ :=
  (coveredEvent k i).indicator 1

lemma measurable_strictRank {ι : Type*} [Fintype ι] (i : ι) :
    Measurable (fun s : ι → ℝ => strictRank s i) := by
  classical
  have hsum : Measurable (fun s : ι → ℝ =>
      ∑ j : ι, if s j < s i then (1 : ℕ) else 0) := by
    apply Finset.measurable_fun_sum
    intro j hj
    exact Measurable.ite
      (measurableSet_lt (measurable_pi_apply j) (measurable_pi_apply i))
      measurable_const measurable_const
  simpa [strictRank] using hsum

lemma measurableSet_covered {ι : Type*} [Fintype ι] (k : ℕ) (i : ι) :
    MeasurableSet (coveredEvent k i) :=
  measurableSet_lt (measurable_strictRank i) measurable_const

lemma integrable_coverInd {ι : Type*} [Fintype ι] (μ : Measure (ι → ℝ))
    [IsProbabilityMeasure μ] (k : ℕ) (i : ι) : Integrable (coverInd k i) μ := by
  exact (integrable_const (1 : ℝ)).indicator (measurableSet_covered k i)

lemma integral_coverInd {ι : Type*} [Fintype ι] (μ : Measure (ι → ℝ))
    (k : ℕ) (i : ι) : ∫ s, coverInd k i s ∂μ = μ.real (coveredEvent k i) :=
  integral_indicator_one (measurableSet_covered k i)

lemma coverInd_eq {ι : Type*} [Fintype ι] (s : ι → ℝ) (k : ℕ) (i : ι) :
    coverInd k i s = if strictRank s i < k then 1 else 0 := by
  classical
  by_cases h : strictRank s i < k <;> simp [coverInd, coveredEvent, h]

lemma coverInd_reindex {ι : Type*} [Fintype ι] (e : Equiv.Perm ι)
    (s : ι → ℝ) (k : ℕ) (i : ι) : coverInd k i (reindex e s) = coverInd k (e i) s := by
  classical
  simp only [coverInd_eq, strictRank_reindex]

lemma integral_coverInd_eq {ι : Type*} [Fintype ι] (μ : Measure (ι → ℝ))
    (hex : ExchangeableLaw μ) (k : ℕ) (i j : ι) :
    ∫ s, coverInd k j s ∂μ = ∫ s, coverInd k i s ∂μ := by
  classical
  have h := (hex (Equiv.swap i j)).integral_comp' (coverInd k i)
  simpa only [coverInd_reindex, Equiv.swap_apply_left] using h

/-- Exchangeability turns the deterministic count into a probability bound. -/
theorem exchangeable_coverage_count {ι : Type*} [Fintype ι]
    (μ : Measure (ι → ℝ)) [IsProbabilityMeasure μ] (hex : ExchangeableLaw μ)
    (k : ℕ) (hk : k ≤ Fintype.card ι) (i : ι) :
    (k : ℝ) ≤ (Fintype.card ι : ℝ) * μ.real (coveredEvent k i) := by
  classical
  have hpoint (s : ι → ℝ) : (k : ℝ) ≤ ∑ j : ι, coverInd k j s := by
    have hcard := card_covered_ge s k hk
    have heq : (∑ j : ι, coverInd k j s) =
        ((Finset.univ.filter fun j => strictRank s j < k).card : ℝ) := by
      simp only [coverInd_eq]
      exact Finset.sum_boole (R := ℝ) (fun j : ι => strictRank s j < k) Finset.univ
    rw [heq]
    exact_mod_cast hcard
  calc
    (k : ℝ) = ∫ _s : ι → ℝ, (k : ℝ) ∂μ := by simp
    _ ≤ ∫ s, ∑ j : ι, coverInd k j s ∂μ :=
      integral_mono (integrable_const _) (integrable_finsetSum _
        (fun j _ => integrable_coverInd μ k j)) hpoint
    _ = ∑ j : ι, ∫ s, coverInd k j s ∂μ :=
      integral_finsetSum _ (fun j _ => integrable_coverInd μ k j)
    _ = ∑ _j : ι, μ.real (coveredEvent k i) := by
      apply Finset.sum_congr rfl
      intro j hj
      exact (integral_coverInd_eq μ hex k i j).trans (integral_coverInd μ k i)
    _ = (Fintype.card ι : ℝ) * μ.real (coveredEvent k i) := by simp

/-- Corrected split-conformal coverage, allowing arbitrary tied scores. -/
theorem split_conformal_rank_coverage (m : ℕ) (μ : Measure (Fin (m + 1) → ℝ))
    [IsProbabilityMeasure μ] (hex : ExchangeableLaw μ) (α : ℝ)
    (hα0 : 0 < α) (_hα1 : α < 1) (test : Fin (m + 1)) :
    1 - α ≤ μ.real (coveredEvent ⌈((m + 1 : ℕ) : ℝ) * (1 - α)⌉₊ test) := by
  let k := ⌈((m + 1 : ℕ) : ℝ) * (1 - α)⌉₊
  change 1 - α ≤ μ.real (coveredEvent k test)
  have hN : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  have hk : k ≤ Fintype.card (Fin (m + 1)) := by
    simp only [Fintype.card_fin]
    apply Nat.ceil_le.mpr
    nlinarith
  have hround : ((m + 1 : ℕ) : ℝ) * (1 - α) ≤ (k : ℝ) := Nat.le_ceil _
  have hcount := exchangeable_coverage_count μ hex k hk test
  simp only [Fintype.card_fin] at hcount
  exact (mul_le_mul_iff_right₀ hN).mp (by simpa [mul_comm] using hround.trans hcount)

/-- Removing the tested coordinate leaves the strict rank unchanged. -/
lemma calibration_strict_count_eq_rank {ι : Type*} [Fintype ι] [DecidableEq ι]
    (s : ι → ℝ) (i : ι) :
    ((Finset.univ.erase i).filter (fun j => s j < s i)).card = strictRank s i := by
  classical
  unfold strictRank
  congr 1
  ext j
  by_cases h : j = i
  · subst j
    simp
  · simp [h]

/-- Exchangeable calibration provides at least nominal marginal coverage.
The threshold is constructed from the calibration coordinates only;
its index is ceil((m+1)(1-alpha)), and an unavailable order statistic is +infinity. -/
theorem split_conformal_coverage (m : ℕ) (μ : Measure (Fin (m + 1) → ℝ))
    [IsProbabilityMeasure μ] (hex : ExchangeableLaw μ) (α : ℝ)
    (hα0 : 0 < α) (hα1 : α < 1) (test : Fin (m + 1)) :
    1 - α ≤ μ.real {s | (s test : WithTop ℝ) ≤
      calibrationThreshold (Finset.univ.erase test) s
        ⌈((m + 1 : ℕ) : ℝ) * (1 - α)⌉₊} := by
  classical
  let k := ⌈((m + 1 : ℕ) : ℝ) * (1 - α)⌉₊
  have hkpos : 0 < k :=
    Nat.ceil_pos.mpr (mul_pos (by positivity) (sub_pos.mpr hα1))
  have hevents : {s : Fin (m + 1) → ℝ | (s test : WithTop ℝ) ≤
      calibrationThreshold (Finset.univ.erase test) s k} = coveredEvent k test := by
    ext s
    change ((s test : WithTop ℝ) ≤ calibrationThreshold (Finset.univ.erase test) s k) ↔
      strictRank s test < k
    rw [le_calibrationThreshold_iff _ _ _ hkpos]
    rw [calibration_strict_count_eq_rank]
  change 1 - α ≤ μ.real {s : Fin (m + 1) → ℝ | (s test : WithTop ℝ) ≤
    calibrationThreshold (Finset.univ.erase test) s k}
  rw [hevents]
  exact split_conformal_rank_coverage m μ hex α hα0 hα1 test

/-- The sharp corrected-index bound and the nominal coverage guarantee, with ties allowed. -/
theorem split_conformal_coverage_sharp (m : ℕ) (μ : Measure (Fin (m + 1) → ℝ))
    [IsProbabilityMeasure μ] (hex : ExchangeableLaw μ) (α : ℝ)
    (hα0 : 0 < α) (hα1 : α < 1) (test : Fin (m + 1)) :
    let k := ⌈((m + 1 : ℕ) : ℝ) * (1 - α)⌉₊
    1 - α ≤ (k : ℝ) / ((m + 1 : ℕ) : ℝ) ∧
      (k : ℝ) / ((m + 1 : ℕ) : ℝ) ≤ μ.real {s | (s test : WithTop ℝ) ≤
        calibrationThreshold (Finset.univ.erase test) s k} := by
  classical
  let k := ⌈((m + 1 : ℕ) : ℝ) * (1 - α)⌉₊
  change 1 - α ≤ (k : ℝ) / ((m + 1 : ℕ) : ℝ) ∧
    (k : ℝ) / ((m + 1 : ℕ) : ℝ) ≤ μ.real {s | (s test : WithTop ℝ) ≤
      calibrationThreshold (Finset.univ.erase test) s k}
  have hN : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  have hkpos : 0 < k := Nat.ceil_pos.mpr (mul_pos hN (sub_pos.mpr hα1))
  have hk : k ≤ Fintype.card (Fin (m + 1)) := by
    simp only [Fintype.card_fin]
    apply Nat.ceil_le.mpr
    nlinarith
  have hevents : {s : Fin (m + 1) → ℝ | (s test : WithTop ℝ) ≤
      calibrationThreshold (Finset.univ.erase test) s k} = coveredEvent k test := by
    ext s
    change ((s test : WithTop ℝ) ≤ calibrationThreshold (Finset.univ.erase test) s k) ↔
      strictRank s test < k
    rw [le_calibrationThreshold_iff _ _ _ hkpos]
    rw [calibration_strict_count_eq_rank]
  constructor
  · apply (le_div_iff₀ hN).mpr
    simpa only [mul_comm] using
      (Nat.le_ceil (((m + 1 : ℕ) : ℝ) * (1 - α)))
  · apply (div_le_iff₀ hN).mpr
    have hcount := exchangeable_coverage_count μ hex k hk test
    rw [← hevents] at hcount
    simpa only [Fintype.card_fin, mul_comm] using hcount

end Grail.Conformal
