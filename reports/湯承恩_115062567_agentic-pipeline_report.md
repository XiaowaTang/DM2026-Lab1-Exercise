# Agentic Pipeline Report

115062567 湯承恩


I do some data mining on the dataset "20-newsgroups", which contains articles that can be split into 4 categories based on their topics: "soc.religion.christian" (599 documents), "sci.med" (594 documents), "comp.graphics" (584 documents), and "alt.atheism" (480 documents). I sample 2000 documents for the following anaylsis, below are the distribution of document text length of each categories:

![describe result](../plots/湯承恩_agentic-pipeline_plot_describe_7.png)

First, I inspect feature correlation matrix to see how terms with highest variance correlate with each other, below shows the result. It is evident that many of the words with the highest variance are stop-words that carry meaningless information. This is to be expected, as I did not remove them when constructing the document-term matrix. Nevertheless, the matrix still yields some useful insights: for instance, we can observe a high correlation coefficient (0.62) between the terms "jpeg" and "image", a value much higher than the correlation of "jpeg" with stop words such as "the" (0.30). This is likely due to the fact that they frequently appear together in articles within the "comp.graphics" category.

![feat_corr result](../plots/湯承恩_agentic-pipeline_plot_feat_corr_15.png)

To further examine how each terms correlate with each categories, I apply filters such as pearson_filter and spearman_filter on the dataset. The bar chart belows shows the terms with highest absolute Spearman correlation with the "soc.religion.christian" articles. We can see that words such has 'jesus' and 'church' are highly correlate with the religion topic, because they are terms frequently used when discussing religion, but are rarely mentioned in other contexts. In comparison, 'nntp(network news transfer protocol)' has negative correlation, since it is a technical term and will rarely be discuss when talking about religion.

![spearman_corr christian_result](../plots/湯承恩_agentic-pipeline_plot_turn114.png)

Next, I do pattern mining. Here's the configuration I choose:
* filtering_method = 'term_frequency'
* algorithm = 'maxfpgrowth'
* min_sup = 15
I have try several combinations, here's why I choose these:
1. For filter_method the word such as 'subject' keep occurring in the pattern I found, this is because the original text come from newsgroups, so it looks like this: "From: sd345@city.ac.uk (Michael Collier) Subject: Converting images to HP LaserJet III?...". Terms like "from" and "subject" are merely artifacts of the dataset's specific format and carry no meaningful information. Therefore, I have found that the best way to filter out such words is by using variance filtering (which removes terms above the line of 95th percentile).
2. Why I choose 'maxfpgrowth' is because the patterns it find are structurally more concise than standard frequent patterns.
3. I try different min_sup, and setting it to 15 gives the moderate amount of patterns.

To see how this is useful, lets take the result of doing pattern mining to "alt.atheism" articles as an example. The words such as 'marriage' and 'contradiction' shows inside the pattern, I suppose that this is because atheists often employ logical positivism or critiques of secular conventions such as marriage to challenge theists, so they are important patterns of the discussion. However, I noticed a strange pattern in the summary provided by the agent: [private, cheers, activities, alink, ksand, bu...], this pattern with multiple entries likely indicates that there exist a same text that has been forwarded by multiple articles, therefore this may suggest that the forum contains some frequently cited documents.

Finally, I apply 3 different ways of dimensionality reduction (PCA, t-SNE and UMAP) on the dataset to see how articles in different categories group together. Below shows the result of 3 different tries. We can see that PCA is ineffective at separating data from different categories because linear projection cannot handle complex data. In contrast, t-SNE and UMAP employ non-linear techniques, which allow for better separation between different categories.

![pca_result](../plots/湯承恩_agentic-pipeline_plot_reduce_dimensions_36.png)
![t_sne_result](../plots/湯承恩_agentic-pipeline_plot_reduce_dimensions_37.png)
![umap_result](../plots/湯承恩_agentic-pipeline_plot_reduce_dimensions_38.png)