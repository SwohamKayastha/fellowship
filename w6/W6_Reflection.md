# Week 6 Reflection — Probabilistic Models & Bayesian Inference
### Fusemachines AI Fellowship · Statistical Machine Learning

---

*Give one concrete example from this assignment where the fully Bayesian answer would have changed a decision you might have made using only the MLE or a p-value. Explain the mechanism — was it prior information, uncertainty quantification, sequential updating, or the Bayesian-vs-frequentist comparison?*

---

## The Decision: Should We Act on the New Contract Tier?

The VP's second concern in this assignment was precisely about a new contract tier with only **40 customers enrolled** , this is Group A_small from Part 1, and it is the clearest example where the Bayesian answer would have changed an operational decision.

**What the MLE says:** With 15 churns out of 40 customers, the MLE for the churn rate is 15/40 = **0.375**. A product manager using only this number would see a churn rate well above the 0.25 company wide threshold and might recommend an immediate, expensive retention intervention for this new segment.

**What the frequentist p-value adds but not enough:** Running a one proportion z-test against the null hypothesis H₀: θ = 0.265 (overall churn rate) with n=40 gives a p-value of approximately 0.21, not significant at α=0.05. So the frequentist analysis would say "we cannot reject that this segment is the same as the baseline." But this is a purely binary answer: it doesn't tell the VP *how uncertain* the estimate actually is, and it doesn't incorporate any prior knowledge.

**What the full Bayesian posterior reveals:** Using a Beta(2, 8) prior which encodes the domain belief that most segments churn at less than 30%, the posterior for Group A_small is Beta(17, 33). The 94% HDI for this posterior spans approximately **[0.22, 0.47]**. This directly answers the VP's question in a way neither the MLE nor the p-value can: "The 94% credible interval for this segment's true churn rate is [22%, 47%] it could be lower than the overall rate or nearly twice as high."

**The mechanism: uncertainty quantification + prior information combined.** The Bayesian answer changed the decision in two ways. First, the prior pulled the MAP estimate down from 0.375 to 0.333 the Beta(2, 8) prior's 10 pseudocounts represent 25% of the effective sample at n=40, meaningfully regularising the estimate. This prior pull of **0.042** is over 4× larger than the pull for the full Monthto month group (0.0006), where 3,875 observations make the prior irrelevant. Second, and more importantly, the posterior's width tells the VP not just a single number but the full range of plausible values. A VP making a resource allocation decision can see that the lower bound of the HDI (0.22) is actually *below* the company wide churn rate meaning an expensive targeted intervention might be premature until more data arrives.

**The sequential updating connection (Part 2):** The Bayesian framework also told us *exactly when* the evidence would be sufficient to act. The sequential decision boundary analysis (Q6) showed that P(θ > 0.25) first exceeds 0.90 at **n=17 observations**  compared to the frequentist z test which would require **n=6,304** for 80% power. This is not because the Bayesian approach uses less data to reach a valid conclusion, but because the prior encodes existing knowledge that the z test discards. For the VP's 40 customer segment, we can say with 90%+ probability that the churn rate exceeds 25% ,a statement that the frequentist analysis would refuse to make without thousands of additional customers.

**Bottom line:** The MLE alone (0.375) would likely trigger an over confident intervention. The p value alone would give a non significant result that might lead to inaction. The full posterior reporting the HDI [0.22, 0.47] and the sequential evidence accumulation gave the VP precisely what they asked for: not a point estimate, but a principled uncertainty quantification that enables a calibrated decision with only 40 data points.

---

*Fusemachines AI Fellowship · Week 6 · Statistical Machine Learning*
