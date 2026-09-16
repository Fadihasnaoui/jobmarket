"""Cautious-scoring adjustment for recommendations built from a low-confidence CV profile.

Blocking matching entirely on low extraction confidence produced a bad user outcome: a
student or candidate with little formal experience but real skills/projects got zero
recommendations. This module keeps `cv/matching.py`'s scoring untouched and instead
dampens the already-scored recommendations afterward, tagging them with an explicit,
explainable penalty so low confidence becomes a visible warning plus a cautious match
rather than an empty result.
"""

from __future__ import annotations

from jobmarket.cv.profile import CvProfile, JobRecommendation, RecommendationPenalty

LOW_CONFIDENCE_CAUTION_PENALTY = 0.15


def apply_confidence_caution(
    recommendations: list[JobRecommendation],
    profile: CvProfile,
) -> list[JobRecommendation]:
    """Dampen and tag recommendations when CV extraction confidence is low.

    No-op when confidence isn't low or there is nothing to adjust, so normal-confidence
    profiles are scored exactly as `recommend_jobs_for_cv` returned them.
    """
    if not recommendations or profile.extraction_quality.extraction_confidence_label != "low":
        return recommendations
    score = profile.extraction_quality.extraction_confidence_score
    penalty = RecommendationPenalty(
        code="low_extraction_confidence",
        amount=LOW_CONFIDENCE_CAUTION_PENALTY,
        explanation=(
            f"CV extraction confidence is low (score={score:.2f}); score reduced "
            "cautiously rather than withheld."
        ),
    )
    adjusted = [
        rec.model_copy(
            update={
                "final_score": round(
                    max(0.0, rec.final_score / 100 - LOW_CONFIDENCE_CAUTION_PENALTY) * 100,
                    2,
                ),
                "penalties": [*rec.penalties, penalty],
            }
        )
        for rec in recommendations
    ]
    adjusted.sort(key=lambda item: (-item.final_score, item.job_id))
    return adjusted


__all__ = ["LOW_CONFIDENCE_CAUTION_PENALTY", "apply_confidence_caution"]
