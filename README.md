# Project 2 - Ames Housing Data and Kaggle Challenge

## Executive Summary

### Problem Statement

Using the Ames Housing Dataset that is available on [Kaggle](https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge), we want to identify which features are the best predictors of housing price and create a regression model that will help us make predictions with the best R<sup>2</sup> score.

The Kaggle challenge offers a train.csv for us to to train our model with, and a test.csv which we will clean and fit the model onto, in order to make our predictions. A csv of the predictions is ultimately uploaded to the Kaggle challenge for scoring.

The model will be tuned closely to the Ames Housing dataset, and we might be able to use our findings from the process to understand what are some key predictors we can use in predicting prices for houses in the United States. However, given that this data set has some features that are very specific to Ames, Iowa (e.g. neighborhood), it will not be perfect fit for other housing data in the U.S.

The model, and our understanding of the key features will be beneficial to existing home owners who might be considering selling their property to have a gauge of what prices their property could fetch. Of course, it will also be highly useful for real estate agents who wish to help their selling customers arrive at an appropriate calling price. Potential home buyers might also find it useful in terms of having a real estimate at what price a house they're looking at might be worth, to adjust their bid price accordingly.


### How it was carried out

#### Data Cleaning and EDA

train.csv contains a whopping 2051 rows and 81 columns so there's a lot we need to do to understand the data at hand.

After the initial import into the dataframe, I started by generating a list of percentages of null values in each feature. This helps to give a broad overview of which features have so many missing values that it would no be useful for modelling. Features with more than 90% null values were discarded.

Next, I ran a describe() on the data set, followed by value_counts(dropna=False) to give a rough sense of the distribution for each non-numeric feature. The latter allowed me to decide if a feature should ultimately be nominal or ordinal. During the process, I recorded in the notebook how many rows of null and 0 or "None" values each feature has, while also cross-referencing the data dictionary provided on Kaggle.

Since we often have sets of related features - e.g. "Mas Vnr Type" (masonry veneer type) and "Mas Vnr Area" - this quick look-through helped me gain a sense of what's causing our NaN/"None"/0 values. For instance, we can infer from the data dictionary that if "Mas Vnr Type" is NaN or "None", it would make sense for "Mas Vnr Area" to have 0 or NaN. We can also infer again that in such cases, NaN values are not the result of errors in data gathering, but rather they reflect real-world circumstances (i.e. no masonry veneer). Comparing the number of rows with NaN/"None"/0 for related features also gives us a sense of how much data processing we'll need to do - would it be a simple fillna()? Or will we need to actual impute data by inferring trends from other rows for a particular feature?

In cases where a certain feature looked overwhelmingly to be heavily skewed with a narrow variance, such that it's not useful anymore as a predictor, I would also try to investigate, and drop that feature if necessary. For instance, when it looked like one class dominated more than 99% of the categorical feature "Roof Matl", I did a quick barplot to check on my hunch and dropped that column.

There were also a lot of sanity checks required. For instance, making sure that if "Bsmnt Qual" = Nan/"None" (i.e. no basement), all other features pertaining to the basement should also be set to "None"/0.

While most of the cells with NaN were truly meant to be "None" or 0 (i.e. the data is not in fact missing), there were some instances where it looked like there might have been issues with data entry.

For instance, "BsmtFin SF 1" and "BsmtFin SF 2" only indicate sqft area for basement area that's finished, so there's another feature "Bsmt Unf SF" that indicates how many sqft is unfinished. Strangely, about 50 or so rows that were expected to be finished (based on "BsmtFin Type 1" or "BsmtFin Type 2" classes) were 0 for "Bsmt Unf SF". This required deciding whether we should drop the rows or to impute missing values.

"Lot Frontage" was another row that needed a bit more fixing. Based on research, I determined that it's unlikely for houses in the U.S. to not have lot frontage, so I imputed based on median values.

#### Preprocessing and Modeling

By cross-referencing with the data dictionary, and making other assumptions based on external research and real-world knowledge (e.g. I decided "Land Contour" is really about personal preference whereas "Land Slope" is most likely ordinal), I decided which categorical features to map into ordinal features or to use one-hot-encoding on. I also considered whether to run get_dummies() with drop='first', but decided that the model is most likely suitable for Lasso regression (more on that later), in which case it wouldn't be necessary to execute drop='first'.

After cleaning up missing data - and before one-hot-encoding and ordinal mapping - I also ran a heatmap against "Sales Price" to see if there were any features that obviously should be dropped based on weak correlation scores. I also extracted the list of features with absolute correlation score less than 0.3 (quite a conservative threshold) and assessed them using real-world understanding before proceeding to drop.

I also chose to run another round of heatmap (again, against "Sales Price") and a linear regression - before executing get_dummies() just to understand if how my new set of numerical features were faring.

While real-world understanding might lead me to think that something like "Kitchen Qual" would be important in price predictions, it's harder to gauge if another feature like "Kitchen AbvGr"  (i.e. number of kitchens above ground level) would actually impact prices - especially since I'm based in Singapore where even the concept of "Kitchen AbvGr" seems odd! Hence, it is not always possible to use common sense in whittling down the predictor set.

Furthermore, given that the number of features is moderately large (~80) and that a lot of features were likely correlated, I felt that ultimately Lasso regression would be the best because it is able to reduce coefficients for less useful features to zero. To use Lasso Regression, I ran LassoCV first to determine an appropriate alpha.

Scaling, of course, is necessary, given that we have a very mixed bag of features that are ordinal (typically 0 to 5), stated in years, stated in square footage etc. StandardScaler was used for this purpose.

train_test_split was also relied upon to help train the model; the default parameters were used, since those are industry standards that have been extensively researched.

#### Evaluation and Conceptual Understanding

Since Kaggle explicity stated that RMSE (Root Mean Square Error, the lower the better) will be used for grading, I made a point to factor that into my evaluation of the models. 

For the train dataset, using Linear Regression, my model performed quite poorly initially.

````
X_train RMSE is: 26396.60407332863
X_val RMSE is: 787596163735163.0
RMSE score worsened (is higher) for X_val
Overal RMSE is: 393897163032029.7
````

However, after I'd used LassoCV to help me with feature selection, RMSE scores on both the validation set and the overall train set improved (i.e. shrunk). Generally, the validation set has a lower RMSE compared to the overall train set; but as long as the model is not overfitting, that's still fine.

These were my results after selecting only the top 30 features.

````
X_train score is: 0.8687386669695785
X_val score is: 0.8870384594659668
Overall train score is: 0.873217485264655

X_train RMSE is: 28991.34700939357
X_val RMSE is: 26501.649684333162
RMSE score improved (is lower) for X_val
Overal RMSE is: 28389.093451118657
````

#### Prepping the test set

For the test set, I started out by dropping the columns that are no longer in the train set. That way, it makes the EDA for the test set much quicker.

However, keeping in mind that some columns in train were completely 'new' thanks to one-hot-encoding, I made sure to compare substrings. For instance, we want to keep 'land_contour' because it corresponds to 'land_contour_hls' (a result of get_dummies) which came up in our model's top 30 features.

In addition, I also retained a few columns that I'd used for feature-engineering, namely 'bsmt_full_bath', 'total_baths', 'bsmt_half_bath' - for calculating 'total_baths' and 'full_bath'.

#### Predicting with the test set

The cleaned up test set was scaled using the cleaned train set. For modeling, I'd used Lasso with the same optimal alpha derived during the earlier modeling phases with the train set. 

I also tried using Linear Regression for comparison.

*Kaggle scores using Linear Regression*
* Private: 30431.20531
* Public: 31041.80593

*Kaggle scores using Lasso*
* Private: 30541.45008
* Public: 30938.56939

Both seem to fare similarly; Linear Regression was slightly better on the private score, whereas Lasso performed slightly better on the public score.

### Conclusion and Recommendations

Based on the Lasso coefficients, these are the top 10 features that I have determined to be useful. They have the strongest coefficient score which also happen to be positive.

* gr_liv_area
* overall_qual
* neighborhood_nridght (Northridge)
* garage_cars
* kitchen_qual
* neighborhood_stonebr (Stonebrook)
* bldg_type_1fam
* exter_qual
* mas_vnr_area
* bsmt_exposure (the quality, if applicable, of walkout or garden level basement walls)

The 5 features that have the stronges negative coefficient scores are:

* bldg_type_twnhse (Townhouse End Unit)
* bldg_type_twnhs (Townhouse Inside Unit)
* neighborhood_edwards
* land_contour_bnk (banked)
* mas_vnr_type_brkface (brick face)

For property investors, I would recommend 
* investing in houses with a large gross living area, preferably a Single-family Detached type with a spacious garage
* investing houses that are either already of good overall quality (interiors and exteriors), or spending on renovation
* to ensure that the kitchen is inviting and in a good condition - after all, cooking/entertaining is very trendy now
* investing in Northbidge and Stonebrook neighborhoods
