# Homework Report

115062567 湯承恩


In the homework, I have done data mining on a dataset "Reddit-stock-sentiment" to analysis how sentimental information embedded when peolple talking about stock. The dataset is split into three categories based on the semantic meaning:
* 0.0 (Neutral): 423 texts
* -1.0 (Negative): 315 texts
* 1.0 (Positive): 109 texts

Below are the distribution of document text length of each categories. We can see that since most of the text are some short comment on the topics, so overall the length is small. However there're some long comments, probably because some one is analysis or explain something in detail to other people.

![describe result](../plots/湯承恩_homework_plot_describe_5.png)

Next, I built document-term matrix and do some analysis such as inspecting most frquent words and feature correlation matrix. However, since the removal of stop words is not permitted during the construction of DTM, most of the terms present are stop words, which contain very little information.

Nevertheless, I moved on to examine how each terms correlate with each categories using Spearman correlation. Below are the terms with highest absolute correlation with three types of sentiment. For "negative" sentiment, we can see that words such as 'trump', 'trade' and 'americans' highly correlate with it, suggest that the current political situation in the United States has caused some negative sentiment fluctuations in the stock market. As for "neutral" sentiment, most of the top-10 associated terms carry a negative connotation, this suggests that while hardly any words are strongly linked to neutral emotion, there are certainly some terms that would never appear in such an expression (such as 'trump' and 'fuck'). As for "positive" sentiment, the term 'technology' is most closely associated with it, reflecting the benefits that investors have reaped from the rapid progress currently seen in the technology domain.

![spearman_corr neg_result](../plots/湯承恩_homework_plot_turn118.png)
![spearman_corr neutral_result](../plots/湯承恩_homework_plot_turn122.png)
![spearman_corr pos_result](../plots/湯承恩_homework_plot_turn126.png)

Next, I do pattern mining. Here's the configuration I eventually picked:
* filtering_method = 'tfidf'
* algorithm = 'fpgrowth'
* min_sup = 5

I found it necessary to set a smaller min_sup value in order to obtain an moderate number of patterns, this is because the dataset is very small (it even only contains 109 "positive" texts). Despite this, there are some meaningful findings(reference ID: mine_patterns_14~16). For example, `['market'] (support=23)`, `['china'] (support=33)` and `['trump'](support=45)` are the core patterns of "negative" sentiment, accompanied by some co-occurrence patterns such as `['china', 'trump'] (support=9)`, `['market', 'china'] (support=6)`, `['china', 'just'] (support=9)`. These patterns also appears in "neutral" sentiment. This demonstrates that whether investors are analyzing the situation neutrally or reacting negatively, U.S.-China relations, speeches related to Trump, and trade policies have all triggered significant market volatility. 

Finally, I apply 3 different ways of dimensionality reduction (PCA, t-SNE and UMAP) on the dataset to see how articles in different categories group together, below shows the result of 3 different tries. However, I find out that all three of them perform not well in this case. I believe this is because emotion is a complex trait, when mapped onto a two-dimensional space, it may not be easy to represent the differences between distinct emotions. Furthermore, discussions about a special domain naturally share a high degree of overlapping vocabulary (such as specialized terminology frequently used when discussing stocks), which further dilutes the distinctions provided by emotion-related terms.
![pca_result](../plots/湯承恩_homework_plot_reduce_dimensions_17.png)
![t_sne_result](../plots/湯承恩_homework_plot_reduce_dimensions_18.png)
![umap_result](../plots/湯承恩_homework_plot_reduce_dimensions_21.png)

