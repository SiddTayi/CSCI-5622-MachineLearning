#| echo : false 
library(arm)
library(arules)
library(arulesViz)
library(viridis)
library(arules)
library(TSP)
library(data.table)
library(ggplot2)
library(Matrix)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(stopwords)
library(tokenizers)


# Import the dataset
# Load data
setwd("D:/Masters-2023/Machine Learning")
news_data <- read.csv("Youtube_news.csv")

# head(news_data)

# Headline tab only
df <- news_data$Headline
head(df)

new_df <- data.frame(Headline = df)

# head(new_df)

sampled_data = sample_n(new_df, 100)

# dim(sampled_data)
# head(sampled_data)

# ----------------------------------------------------------------------------------------------------------#
# DATA CLEANING and PRE PROCESSING
# Create an empty csv file to store the tokenized headlines
transaction_news = "all_news_cleaned_tokenized.csv"

# Open the file
trans <- file(transaction_news)

# Tokenize the text in the "headlines" column
Tokens <- tokenizers::tokenize_words(
  sampled_data$Headline[1], stopwords = stopwords::stopwords("en"), 
  lowercase = T,  strip_punct = T, strip_numeric = T,
  simplify = T)

# Words to be removed
words_to_remove <- c("fuck", "bitch", "shit", "hell", "n", "que", "de", "en", "la", "el", "canci", "porn", "el",	"que", "est", "leyendo", "este", "comentario",	"pasen",	"por",	"mi", "canal",	"que",	"estar",	"subiendo",	"mucho",	"contenidos",	"gracias",	"por",	"detenerte",	"leer",	"este",	"comentario",	"que",	"dios",	'te',	"bendiga", "fucked"
)

# Remove specified words from tokens
Tokens <- Tokens[!Tokens %in% words_to_remove]

# Write tokens (excluding specified words) to the file. It concatenates the tokens into a single string, separates them by commas, and writes them to the file.
#cat(paste(unlist(Tokens), collapse = ","), "\n", file = Trans)

# Append remaining lists of tokens into file
Trans <- file(transaction_news, open = "a")
for (i in 2:nrow(sampled_data)) {
  Tokens <- tokenize_words(sampled_data$Headline[i],
                           stopwords = stopwords::stopwords("en"),
                           lowercase = TRUE,
                           strip_punct = TRUE,
                           simplify = TRUE)
  
  # Remove specified words from tokens
  Tokens <- Tokens[!Tokens %in% words_to_remove]
  
  # Write tokens (excluding specified words) to the file
  cat(paste(unlist(Tokens), collapse = ","), "\n", file = Trans)
  #cat(unlist(Tokens))
}

close(Trans) # Close the file

print("Done processing")

# ----------------------------------------------------------------------------------------------------------#

# Importing Transactional data
news_df <- read.csv(transaction_news, header = FALSE, sep = ",")
head(news_df)

# Convert all columns to char 
news_df <- news_df %>%
  mutate_all(as.character)

(str(news_df))

# Display the dataframe
news_df <- news_df[-c(1), ]
# head(news_df)


NewsTrans = read.transactions("all_news_cleaned_tokenized.csv",
                            #"KumarGroceriesAS_Transactions.csv",
                           rm.duplicates = FALSE, 
                           format = "basket",  ##if you use "single" also use cols=c(1,2)
                           sep="," #) #,  ## csv file
                           ,cols=1
                           ) ## The dataset HAS row numbers


# inspect(NewsTrans)
# ----------------------------------------------------------------------------------------------------------#

# Apriori 
associa_rules <- apriori(data = NewsTrans, 
                        parameter = list(support = 0.04, 
                                         confidence = 0.09, minlen = 2))

                      
# Visualising the results
supp <- inspect(sort(associa_rules, by = 'support')[1:15])
conf <- inspect(sort(associa_rules, by = 'confidence')[1:15])
lift <- inspect(sort(associa_rules, by = 'lift')[1:15])

# Frequency Plot
itemFrequencyPlot(NewsTrans, topN = 15,
                  col = 'lightblue',
                  main = 'Relative Item Frequency Plot')


plot(associa_rules, 
     method = "graph", 
     measure = "confidence", 
     shading = "lift")

plot(associa_rules, 
     method = "graph", 
     measure = "support", 
     shading = "lift")


# Use apriori to get the RULES
FrulesK = arules::apriori(NewsTrans, parameter = list(support = 0.03, 
                                         confidence = 0.05, minlen = 2))

# Visualising the results
supp <- inspect(sort(FrulesK, by = 'support')[1:15])
conf <- inspect(sort(FrulesK, by = 'confidence')[1:15])
lift <- inspect(sort(FrulesK, by = 'lift')[1:15])


# FREQUENCY PLOT
itemFrequencyPlot(CommentTrans, topN=25, type="absolute")

## Sort rules by a measure such as conf, sup, or lift
SortedRulesK <- sort(FrulesK, by="lift", decreasing=FALSE)
inspect(SortedRulesK[1:5])
(summary(SortedRulesK))


subrulesK <- head(sort(SortedRulesK, by="lift"),20)
plot(subrulesK)


# 3d plot
plot(subrulesK, method="graph", engine="interactive")
