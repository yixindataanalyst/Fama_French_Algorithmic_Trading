# Algorithmic Trading with the Fama-French Three and Five-Factor Models
![Algo](https://cdn.corporatefinanceinstitute.com/assets/fama-french-three-factor-model02.png)

This project set out to examine the Fama-French Three and Five-Factor Models in order to determine which model resulted in the greatest profit.  First, a brief background on the Fama-French model (information obtained from https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp): 

The first model, the Three-Factor Model, is an asset pricing model developed in 1992 by Nobel laureates Eugene Fame and Kenneth French.  It expands on the capital asset pricing model (CAPM) by adding size, risk and value factors to the already established market-risk factor in CAPM.  According to the Investopedia article mentioned above, "this model considers the fact that value and small-cap stocks outperform markets on a regular basis.  By including these two additional factors, the model adjusts for this outperforming tendency, which is thought to make it a better tool for evaluating manager performance."  The three factors are: SMB (Small Minus Big returns), HML (High Minus Low returns) and the portfolio's return minus the risk free rate of return.  

In 2014, the duo expanded their model to a 5-Factor Model, adding RMB (Robust Minus Weak returns), a profitability factor, and CMA (Conservative Minus Aggressive returns), an investment factor.  In theory, these 2 additional factors should have a beneficial impact on being able to make predictions.  Yet, as we will see, there are interesting results when comparing the 3-Factor and 5-Factor returns.

The respective factors for each model were used as features in a Machine Learning model, predictions were generated, and results were evaluated to determine which model, the 3-Factor or the 5-Factor model, was more effective on which to base trading signals.  The following ReadMe will depict the results of ATT's stock for the sake of brevity.  To see full results of $DIS and $SPY as well, please see "Screenshots" folder.

Please note: This project was created in Google Colab to accomodate for the group, and all code is meant to be run within a Google Colab notebook.

Team members include Ben Fischler, Zachary Green, Priya Roy, Frank Xu and Emmanuel Henao.

---

## Pre-Processing:
## Fama-French Three-Factor Model:
In order to more seamlessly progress through the project, we defined various functions that would allow the team to easily read in a CSV and return a cleaned DataFrame:

    def get_factors(factors):
      factor_file = factors + ".csv"
      factor_df = pd.read_csv(factor_file)
      factor_df = factor_df.rename(columns={'Unnamed: 0': 'Date'})

      factor_df['Date'] = factor_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
      factor_df = factor_df.set_index('Date')

      return factor_df
      
With this "get_factors" function, we were then able to quickly read in any dataframe and not have to manually clean it each time.  Using this function resulted in the following Fama-French Three-Factor dataframe from which we were able to work off:

![three_factor](/Screenshots/three_factor.png?raw=true)

The next function we defined allowed us to the very same thing, but this time, with numbers related to a specific stock's returns as opposed to Fama-French factors:

    def choose_stock(ticker):
      ticker_file=ticker+".csv"
      stock=pd.read_csv(ticker_file, index_col='Date', parse_dates=True, infer_datetime_format=True)
      stock["Returns"]=stock["Close"].dropna().pct_change()*100
      stock.index = pd.Series(stock.index).dt.date

      return stock
      
With this "choose_stock" function, we were then able to quickly read in any dataframe and not have to manually clean it each time.  Using this function resulted in the following ATT dataframe from which we were able to work off:

![att](/Screenshots/att.png?raw=true)

Next, we concatenated the above two dataframes so that we had all of the factors we wanted to use (the 3 Fama-French factors), as well as the target column (Returns) we wanted to use, in the same dataframe:

    combined_df = pd.concat([factors, stock], axis='columns', join='inner')
    combined_df = combined_df.dropna()
    combined_df = combined_df.drop('RF', axis=1)

This gave us the following DataFrame: Mkt-RF, SMB and HML as the 3 factors, and Returns as the target:

![concat_df](/Screenshots/concat_df.png?raw=true)

We were now prepared to define X/y variables, split into Training/Testing data, and run our Linear Regression using SK Learn:

    from sklearn.linear_model import LinearRegression
    lin_reg_model = LinearRegression(fit_intercept=True)
    lin_reg_model = lin_reg_model.fit(X_train, y_train)
    predictions = lin_reg_model.predict(X_test)

We converted y_test to a dataframe using ".to_frame()" and added these predictions to the y_test dataframe.  Now came time to generate the signals we would use to backtest our trading strategy, based on the predictions obtained from the 3 Fama-French factors: we decided to "buy" when the day's predicted returns were greater than the day's actual returns, and "sell" when the opposite was true:

    y_test['Buy Signal'] = np.where(y_test['Predictions'] > y_test['Returns'], 1.0,0.0)
    
Now that we had all of this information together in one dataframe, we were able to build on what was already in there, adding new columns showing portfolio performance as time went on.  Again, we defined a function to make it easier to change dataframe working off of, starting capital and share count:

    def generate_signals(input_df, start_capital=100000, share_count=2000):
      initial_capital = float(start_capital)
      signals_df = input_df.copy()
      share_size = share_count

      # Take a 500 share position where the Buy Signal is 1 (day's predictions greater than day's returns):
      signals_df['Position'] = share_size * signals_df['Buy Signal']

      # Make Entry / Exit Column:
      signals_df['Entry/Exit']=signals_df["Buy Signal"].diff()

      # Find the points in time where a 500 share position is bought or sold:
      signals_df['Entry/Exit Position'] = signals_df['Position'].diff()

      # Multiply share price by entry/exit positions and get the cumulative sum:
      signals_df['Portfolio Holdings'] = signals_df['Close'] * signals_df['Entry/Exit Position'].cumsum()

      # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio:
      signals_df['Portfolio Cash'] = initial_capital - (signals_df['Close'] * signals_df['Entry/Exit Position']).cumsum()

      # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments):
      signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']

      # Calculate the portfolio daily returns:
      signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()

      # Calculate the cumulative returns:
      signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1

      signals_df = signals_df.dropna()
  
      return signals_df

Using this function, we generated our backtested portfolio returns for y_test.  This resulted in the following dataframe, which shows important numbers like Portfolio Holdings and Cumulative Portfolio Returns:

![signals_df](/Screenshots/signals_df.png?raw=true)

Next came time to evaluate the algorithm, and again, we utiliized a custom function to easily calculate different metrics based on which dataframe is read-in:

    def algo_evaluation(signals_df):
      # Prepare DataFrame for metrics
      metrics = ['Annual Return', 'Cumulative Returns', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio']

      columns = ['Backtest']

      # Initialize the DataFrame with index set to evaluation metrics and column as `Backtest`:
      portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
      
      # Calculate cumulative returns:
      portfolio_evaluation_df.loc['Cumulative Returns'] = signals_df['Portfolio Cumulative Returns'][-1]
      
      # Calculate annualized returns:
      portfolio_evaluation_df.loc['Annual Return'] = (signals_df['Portfolio Daily Returns'].mean() * 252)
      
      # Calculate annual volatility:
      portfolio_evaluation_df.loc['Annual Volatility'] = (signals_df['Portfolio Daily Returns'].std() * np.sqrt(252))
      
      # Calculate Sharpe Ratio:
      portfolio_evaluation_df.loc['Sharpe Ratio'] = (signals_df['Portfolio Daily Returns'].mean() * 252) / (signals_df['Portfolio Daily Returns'].std() * np.sqrt(252))

      #Calculate Sortino Ratio/Downside Return:
      sortino_ratio_df = signals_df[['Portfolio Daily Returns']].copy()
      sortino_ratio_df.loc[:,'Downside Returns'] = 0
      target = 0
      mask = sortino_ratio_df['Portfolio Daily Returns'] < target
      sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Portfolio Daily Returns']**2
      down_stdev = np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
      expected_return = sortino_ratio_df['Portfolio Daily Returns'].mean() * 252
      sortino_ratio = expected_return/down_stdev
      portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio

      return portfolio_evaluation_df

Using this function, we were able to generate an evaluation table for any dataframe we read-in.  This is the one for ATT, based on the signals dataframe generated above:

![metrics](/Screenshots/metrics.png?raw=true)

Then we generated the same metrics, but this time, without using our backtested strategy of buy-and-sell; this time, we generated the metrics based on a buy-and-hold strategy to compare how our model did vs. how it would have done should it have simply held throughout:

![metrics_with_underlying](/Screenshots/metrics_with_underlying.png?raw=true)

Last step was to define a function which accepted the daily signals dataframe and returned evaluations of individual trades:

    def trade_evaluation(signals_df):
      trade_evaluation_df = pd.DataFrame(
        columns=['Entry Date', 'Exit Date', 'Shares', 'Entry Share Price', 'Exit Share Price', 'Entry Portfolio Holding', 'Exit Portfolio Holding', 'Profit/Loss'])
  
      entry_date = ''
      exit_date = ''
      entry_portfolio_holding = 0
      exit_portfolio_holding = 0
      share_size = 0
      entry_share_price = 0
      exit_share_price = 0

      # Loop through signal DataFrame. If `Entry/Exit` is 1, set entry trade metrics.  Else if `Entry/Exit` is -1, set exit trade metrics and calculate profit, then append the record to the trade evaluation DataFramefor index, row in signals_df.iterrows():
          if row['Entry/Exit'] == 1:
              entry_date = index
              entry_portfolio_holding = row['Portfolio Total']
              share_size = row['Entry/Exit Position']
              entry_share_price = row['Close']

          elif row['Entry/Exit'] == -1:
              exit_date = index
              exit_portfolio_holding = abs(row['Portfolio Total'])
              exit_share_price = row['Close']
              profit_loss = exit_portfolio_holding - entry_portfolio_holding
              trade_evaluation_df = trade_evaluation_df.append(
                  {
                      'Entry Date': entry_date, 'Exit Date': exit_date, 'Shares': share_size, 'Entry Share Price': entry_share_price, 'Exit Share Price': exit_share_price, 'Entry Portfolio Holding': entry_portfolio_holding, 'Exit Portfolio Holding': exit_portfolio_holding, 'Profit/Loss': profit_loss}, ignore_index=True)

      return trade_evaluation_df
      
This allowed us to generate the following evaluation dataframe by feeding in the above-generated signals_df.  It calculates important numbers such as Entry/Exit Share Prices and Profit/Loss, all per trade:

![final_metrics](/Screenshots/final_metrics.png?raw=true)
      
### 3-Factor Results (for ATT):
ANOVA tables were generated utilizing statsmodels' Ordinary Least Squares (OLS) model, a model analogous to Linear Regressions.  The following is the ANOVA table generated for ATT's 3-Factor results:
![att_3_factor_OLS](/Screenshots/att_3_factor_OLS.png?raw=true)

Few of the important numbers we'd like to point out: the first is R-squared.  The way to read this is that 45.7% of the variation in y, which was excess return on the market, is explained by the factors.  As there were only 3 factors, we expected to see that the 5 factor fama French model might yield better results.  

The prob (f-statistic) depicts probability of the null hypothesis being true, and  can be thought of as the p-value for the regression as a whole.  Our f-statistic of near-0 implies that overall, the regressions were meaningful.   

Last thing are the coefficients and the p-values for the X variables.  The coefficients tell you the size of the effect that variable is having on the dependent variable, when all other independent variables are held constant.  We can see out of all 3 variables, Mkt-Rf has the greatest impact on Returns, which makes intuitive sense as Mkt-Rf is directly used to calculate Returns.  In regards to the p-values for the individual variables: these variables are correlated, and we see that a couple of the variables were deemed insignificant by the model, while the overall p-value was significant.  We reasoned that this is because there is multicollinearity going on with the variables that are explaining the same part of the variation in returns, so their "significance" is divided up among them.

Seaborn Pair Plots were also generated for each stock.  This plot shows the affect of all X variables when all of the other X variables are held constant.  It acts as a visualization of the variable coefficients: the higher the coefficient, the more severe the slope of the line will be, either positively or negatively correlated.  This is the Seaborn Pair Plot for ATT's Three-Factor model:
![att_3_factor_plot](/Screenshots/att_3_factor_plot.png?raw=true)
With Mkt-Rf on the X-axis, we see that as it goes up so do returns, which makes sense, as it had the highest coeffiecient.  The other two variables had slight effects on the dependent variable.

![att_3_factor_plot_two](/Screenshots/att_3_factor_plot_two.png?raw=true)
      
The above plot was generated for each stock analyzed: ATT, DIS and SPY, this one being for ATT.  It is a visual representation of the 3-factor algorithm's cumulative returns vs. the underlying, aka buy-and-hold, return.  We see in the above plot that for ATT, our backtested strategy signficantly outperformed a simple buy-and-hold strategy; while our backtested strategy had positive returns, if you were to buy-and-hold, you would have lost money.    
      
---

## Fama-French Five-Factor Model:
Now it was time to do the same as we just did, but this time, using the Fama-French Five-Factor Model rather than the Three-Factor model.  The 2 additional factors are RMW (Robust Minus Weak returns, aka the Profitability Factor) and CMA (Conservative Minus Aggressive returns, aka the Investment Factor).  Given that there were 2 additional factors added for which the model could utilize to help make predictions, the group anticipated that these results would be significantly better than those we had observed for the Three-Factor model.

This is where our custom functions came in handy: we used our pre-defined functions to read-in the new Five-Factor dataframe and perform all the necessary calculations and evaluations.  New code was not needed to be written.  Therefore, we can jump straight to results for the Five-Factor Model:

### 5-Factor Results (for ATT):
Just like for the Three-Factor Model, ANOVA tables were generated utilizing statsmodels' Ordinary Least Squares (OLS) model, a model analogous to Linear Regressions.  The following is the ANOVA table generated for ATT's 5-Factor results:
![att_5_factor_OLS](/Screenshots/att_5_factor_OLS.png?raw=true)

R-squared increased only slightly, from 45.7% to 47.3%, and the f-statistic was already near-0 with 3 factor.  In terms of coefficients and individual p-values, Mkt-Rf remained the most influential variable, while the one of the new variables, RMW, was the 2nd most influential.  Again, we saw multicollinearity among the X-variables.  Overall, adding these 2 additional factors did not have a significant impact on the model.

![att_5_factor_plot](/Screenshots/att_5_factor_plot.png?raw=true)
Similar to the Three-Factor model, we see that Mkt-Rf is the most influential stock in either direction, positive or negative.  With more factors added, these charts were less steep than when there are only 3 factors because with more relevant factors added, each one now inherently has less of an effect on returns.

![att_5_factor_plot_two](/Screenshots/att_5_factor_plot_two.png?raw=true)
      
We see in the above plot that for ATT, our backtested strategy signficantly outperformed a simple buy-and-hold strategy; while our backtested strategy had positive returns, if you were to buy-and-hold, you would have lost money.  Interestingly, total cumulative returns for the Five-Factor Model were less than those of the Three-Factor Model.  
      
---

## Full Results/Implications/Conclusions:
### For ANOVA tables/graphs of $DIS and $SPY, please see "Screenshots" folder
$T: 45.7% R-squared for 3-factor, 47.3% R-squared for 5-factor; .0978 Cumulative Return for 3-factor, .0688 Cumulative Return for 5-factor  
$DIS: 51.2% R-squared for 3-factor, 52.2% R-squared for 5-factor; -.416 Cumulative Return for 3-factor, -.3166 Cumulative Return for 5-factor  
$SPY: 99.3% R-squared for 3-factor, 99.3% R-squared for 5-factor; .5308 Cumulative Return for 3-factor, .4516 Cumulative Return for 5-factor  

As the above shows, the 5-factor model did not always improve the model / increase our hypothetical, backtested returns.  In terms of R-squared, the 5-factor model only *slightly* improved the values for ATT and Disney, and for SPY, it did not improve as it was already at 99.3%.  In terms of the algorithm's cumulative returns, overall, adding the 2 additional factors actually decreased overall returns; only when the return was negative, like it was for $DIS, did adding the 2 additional factors benefit by decreasing overall loss.

This is not something that the group was necessarily surprised by, especiallly after doing our research and reading about what economists really thought about the validity of the 2 models.  Ultimately, the 5-factor model still ignores momentum and volatility factors, 2 factors believed to have an important impact on a stock's price, and instead opts for adding profitability and investment factors (RMW and CMA, respectively).  Many experts seem to question why these two particular factors were added to the 5-Factor model, and not a more appropriate set of variables; why stop at adding 2?  Perhaps a next-step to this project would be to include these missing factors, momentum and volatility, and see if those factors are more reliable than profitability and investment factors.

In conclusion, we hope to have given a clear application of the Fama-French 3 and 5-Factor Model and how it can be applied to algorithmic trading / trading signals.  Our results mirrored the findings of previous research in this area: the Fama-French 5-Factor model, although introducing 2 new factors that the 3-Factor Model does not have, is not necessarily a more effective model on which to make investment decisions.u s a g e :   g i t   [ - v   |   - - v e r s i o n ]   [ - h   |   - - h e l p ]   [ - C   < p a t h > ]   [ - c   < n a m e > = < v a l u e > ]  
                       [ - - e x e c - p a t h [ = < p a t h > ] ]   [ - - h t m l - p a t h ]   [ - - m a n - p a t h ]   [ - - i n f o - p a t h ]  
                       [ - p   |   - - p a g i n a t e   |   - P   |   - - n o - p a g e r ]   [ - - n o - r e p l a c e - o b j e c t s ]   [ - - b a r e ]  
                       [ - - g i t - d i r = < p a t h > ]   [ - - w o r k - t r e e = < p a t h > ]   [ - - n a m e s p a c e = < n a m e > ]  
                       [ - - s u p e r - p r e f i x = < p a t h > ]   [ - - c o n f i g - e n v = < n a m e > = < e n v v a r > ]  
                       < c o m m a n d >   [ < a r g s > ]  
  
 T h e s e   a r e   c o m m o n   G i t   c o m m a n d s   u s e d   i n   v a r i o u s   s i t u a t i o n s :  
  
 s t a r t   a   w o r k i n g   a r e a   ( s e e   a l s o :   g i t   h e l p   t u t o r i a l )  
       c l o n e           C l o n e   a   r e p o s i t o r y   i n t o   a   n e w   d i r e c t o r y  
       i n i t             C r e a t e   a n   e m p t y   G i t   r e p o s i t o r y   o r   r e i n i t i a l i z e   a n   e x i s t i n g   o n e  
  
 w o r k   o n   t h e   c u r r e n t   c h a n g e   ( s e e   a l s o :   g i t   h e l p   e v e r y d a y )  
       a d d               A d d   f i l e   c o n t e n t s   t o   t h e   i n d e x  
       m v                 M o v e   o r   r e n a m e   a   f i l e ,   a   d i r e c t o r y ,   o r   a   s y m l i n k  
       r e s t o r e       R e s t o r e   w o r k i n g   t r e e   f i l e s  
       r m                 R e m o v e   f i l e s   f r o m   t h e   w o r k i n g   t r e e   a n d   f r o m   t h e   i n d e x  
  
 e x a m i n e   t h e   h i s t o r y   a n d   s t a t e   ( s e e   a l s o :   g i t   h e l p   r e v i s i o n s )  
       b i s e c t         U s e   b i n a r y   s e a r c h   t o   f i n d   t h e   c o m m i t   t h a t   i n t r o d u c e d   a   b u g  
       d i f f             S h o w   c h a n g e s   b e t w e e n   c o m m i t s ,   c o m m i t   a n d   w o r k i n g   t r e e ,   e t c  
       g r e p             P r i n t   l i n e s   m a t c h i n g   a   p a t t e r n  
       l o g               S h o w   c o m m i t   l o g s  
       s h o w             S h o w   v a r i o u s   t y p e s   o f   o b j e c t s  
       s t a t u s         S h o w   t h e   w o r k i n g   t r e e   s t a t u s  
  
 g r o w ,   m a r k   a n d   t w e a k   y o u r   c o m m o n   h i s t o r y  
       b r a n c h         L i s t ,   c r e a t e ,   o r   d e l e t e   b r a n c h e s  
       c o m m i t         R e c o r d   c h a n g e s   t o   t h e   r e p o s i t o r y  
       m e r g e           J o i n   t w o   o r   m o r e   d e v e l o p m e n t   h i s t o r i e s   t o g e t h e r  
       r e b a s e         R e a p p l y   c o m m i t s   o n   t o p   o f   a n o t h e r   b a s e   t i p  
       r e s e t           R e s e t   c u r r e n t   H E A D   t o   t h e   s p e c i f i e d   s t a t e  
       s w i t c h         S w i t c h   b r a n c h e s  
       t a g               C r e a t e ,   l i s t ,   d e l e t e   o r   v e r i f y   a   t a g   o b j e c t   s i g n e d   w i t h   G P G  
  
 c o l l a b o r a t e   ( s e e   a l s o :   g i t   h e l p   w o r k f l o w s )  
       f e t c h           D o w n l o a d   o b j e c t s   a n d   r e f s   f r o m   a n o t h e r   r e p o s i t o r y  
       p u l l             F e t c h   f r o m   a n d   i n t e g r a t e   w i t h   a n o t h e r   r e p o s i t o r y   o r   a   l o c a l   b r a n c h  
       p u s h             U p d a t e   r e m o t e   r e f s   a l o n g   w i t h   a s s o c i a t e d   o b j e c t s  
  
 ' g i t   h e l p   - a '   a n d   ' g i t   h e l p   - g '   l i s t   a v a i l a b l e   s u b c o m m a n d s   a n d   s o m e  
 c o n c e p t   g u i d e s .   S e e   ' g i t   h e l p   < c o m m a n d > '   o r   ' g i t   h e l p   < c o n c e p t > '  
 t o   r e a d   a b o u t   a   s p e c i f i c   s u b c o m m a n d   o r   c o n c e p t .  
 S e e   ' g i t   h e l p   g i t '   f o r   a n   o v e r v i e w   o f   t h e   s y s t e m .  
 