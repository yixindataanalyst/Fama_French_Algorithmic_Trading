# Algorithmic Trading with the Fama-French Three and Five-Factor Models

This project set out to examine the Fama-French Three and Five-Factor Models in order to determine which model resulted in the greatest profit.  First, a brief background on the Fama-French model (information obtained from https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp): 

The first model, the Three-Factor Model, is an asset pricing model developed in 1992 by Nobel laureates Eugene Fame and Kenneth French.  It expands on the capital asset pricing model (CAPM) by adding size, risk and value factors to the already established market-risk factor in CAPM.  According to the Investopedia article mentioned above, "this model considers the fact that value and small-cap stocks outperform markets on a regular basis.  By including these two additional factors, the model adjusts for this outperforming tendency, which is thought to make it a better tool for evaluating manager performance."  The three factors are: SMB (Small Minus Big returns), HML (High Minus Low returns) and the portfolio's return minus the risk free rate of return.  

In 2014, the duo expanded their model to a 5-Factor Model, adding RMB (Robust Minus Weak returns), a profitability factor, and CMA (Conservative Minus Aggressive returns), an investment factor.  In theory, these 2 additional factors should have a beneficial impact on being able to make predictions.  Yet, as we will see, there are interesting results when comparing the 3-Factor and 5-Factor returns.

The respective factors for each model were used as features in a Machine Learning model, predictions were generated, and results were evaluated to determine which model, the 3-Factor or the 5-Factor model, was more effective on which to base trading signals. 

## Fama-French Three-Factor Model:
### 3-Factor Results (for ATT):

ANOVA tables were generated utilizing statsmodels' Ordinary Least Squares (OLS) model, a model analogous to Linear Regressions.  The following is the ANOVA table generated for ATT's 3-Factor results:
![att_3_factor_OLS](/Screenshots/att_3_factor_OLS.png?raw=true)

 It is a visual representation of the 3-factor algorithm's cumulative returns vs. the underlying, aka buy-and-hold, return.  We see in the above plot that for ATT, our backtested strategy signficantly outperformed a simple buy-and-hold strategy; while our backtested strategy had positive returns, if you were to buy-and-hold, you would have lost money.
 ![att_3_factor_plot_two](/Screenshots/att_3_factor_plot_two.png?raw=true)

## Fama-French Five-Factor Model:
Just like for the Three-Factor Model, ANOVA tables were generated utilizing statsmodels' Ordinary Least Squares (OLS) model, a model analogous to Linear Regressions.  The following is the ANOVA table generated for ATT's 5-Factor results:
![att_5_factor_OLS](/Screenshots/att_5_factor_OLS.png?raw=true)

We see in the above plot that for ATT, our backtested strategy signficantly outperformed a simple buy-and-hold strategy; while our backtested strategy had positive returns, if you were to buy-and-hold, you would have lost money.  Interestingly, total cumulative returns for the Five-Factor Model were less than those of the Three-Factor Model.  
![att_5_factor_plot_two](/Screenshots/att_5_factor_plot_two.png?raw=true)

