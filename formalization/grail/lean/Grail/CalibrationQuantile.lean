import Mathlib.Data.Finset.Max
import Mathlib.Data.Finset.Card
import Mathlib.Basic.Real.Basic

noncomputable section
namespace Grail.Conformal

/-- Standard kth-order-statistic characterization, valid with repeated scores. -/
def IsCalibrationQuantile {ι : Type*} (cal : Finset ι) (s : ι → ℝ)
    (k : ℕ) (q : ℝ) : Prop :=
  (cal.filter (fun j => s j < q)).card < k ∧
    k ≤ (cal.filter (fun j => s j ≤ q)).card

/-- A finite calibration sample has a kth order statistic whenever 1≤k≤its size. -/
theorem exists_calibration_quantile {ι : Type*} (cal : Finset ι) (s : ι → ℝ)
    (k : ℕ) (hk0 : 0 < k) (hkc : k ≤ cal.card) :
    ∃ q, IsCalibrationQuantile cal s k q := by
  classical
  let eligible := cal.filter (fun i => k ≤ (cal.filter (fun j => s j ≤ s i)).card)
  have hcal : cal.Nonempty := Finset.card_pos.mp (hk0.trans_le hkc)
  have heligible : eligible.Nonempty := by
    obtain ⟨i, hi, hmax⟩ := cal.exists_max_image s hcal
    refine ⟨i, Finset.mem_filter.mpr ⟨hi, ?_⟩⟩
    have heq : cal.filter (fun j => s j ≤ s i) = cal := by
      apply Finset.filter_eq_self.mpr
      exact hmax
    simpa [heq] using hkc
  obtain ⟨i, hi, hmin⟩ := eligible.exists_min_image s heligible
  have hiweak : k ≤ (cal.filter (fun j => s j ≤ s i)).card :=
    (Finset.mem_filter.mp hi).2
  refine ⟨s i, ?_, hiweak⟩
  by_contra hnot
  have hstrict : k ≤ (cal.filter (fun j => s j < s i)).card := Nat.le_of_not_gt hnot
  let lower := cal.filter (fun j => s j < s i)
  have hlower : lower.Nonempty := Finset.card_pos.mp (hk0.trans_le hstrict)
  obtain ⟨j, hj, hmax⟩ := lower.exists_max_image s hlower
  have hjcal : j ∈ cal := (Finset.mem_filter.mp hj).1
  have hjlt : s j < s i := (Finset.mem_filter.mp hj).2
  have hsub : lower ⊆ cal.filter (fun l => s l ≤ s j) := by
    intro l hl
    exact Finset.mem_filter.mpr ⟨(Finset.mem_filter.mp hl).1, hmax l hl⟩
  have hjeligible : j ∈ eligible :=
    Finset.mem_filter.mpr ⟨hjcal, hstrict.trans (Finset.card_le_card hsub)⟩
  exact (not_lt_of_ge (hmin j hjeligible)) hjlt

/-- Any value characterized as the kth statistic gives the usual threshold test. -/
theorem le_quantile_iff_strict_count_lt {ι : Type*} (cal : Finset ι)
    (s : ι → ℝ) (k : ℕ) (q t : ℝ) (hq : IsCalibrationQuantile cal s k q) :
    t ≤ q ↔ (cal.filter (fun j => s j < t)).card < k := by
  classical
  constructor
  · intro ht
    apply lt_of_le_of_lt _ hq.1
    apply Finset.card_le_card
    intro j hj
    exact Finset.mem_filter.mpr ⟨(Finset.mem_filter.mp hj).1,
      lt_of_lt_of_le (Finset.mem_filter.mp hj).2 ht⟩
  · intro ht
    by_contra hn
    have hqt : q < t := lt_of_not_ge hn
    have hsub : cal.filter (fun j => s j ≤ q) ⊆ cal.filter (fun j => s j < t) := by
      intro j hj
      exact Finset.mem_filter.mpr ⟨(Finset.mem_filter.mp hj).1,
        lt_of_le_of_lt (Finset.mem_filter.mp hj).2 hqt⟩
    exact (not_le_of_gt ht) (hq.2.trans (Finset.card_le_card hsub))

/-- The finite kth statistic, or +infinity when its index exceeds the sample size. -/
def calibrationThreshold {ι : Type*} (cal : Finset ι) (s : ι → ℝ)
    (k : ℕ) : WithTop ℝ :=
  if h : 0 < k ∧ k ≤ cal.card then
    (Classical.choose (exists_calibration_quantile cal s k h.1 h.2) : ℝ)
  else ⊤

/-- Exact threshold/rank equivalence, including tied scores and infinite threshold. -/
theorem le_calibrationThreshold_iff {ι : Type*} (cal : Finset ι) (s : ι → ℝ)
    (k : ℕ) (hk : 0 < k) (t : ℝ) :
    (t : WithTop ℝ) ≤ calibrationThreshold cal s k ↔
      (cal.filter (fun j => s j < t)).card < k := by
  classical
  by_cases hkc : k ≤ cal.card
  · have h := Classical.choose_spec (exists_calibration_quantile cal s k hk hkc)
    simpa only [calibrationThreshold, dite_eq_left (show 0 < k ∧ k ≤ cal.card from ⟨hk, hkc⟩), WithTop.coe_le_coe] using
      le_quantile_iff_strict_count_lt cal s k _ t h
  · have hcard : (cal.filter (fun j => s j < t)).card < k :=
      (Finset.card_filter_le _ _).trans_lt (Nat.lt_of_not_ge hkc)
    simp [calibrationThreshold, hkc, hcard]

end Grail.Conformal
