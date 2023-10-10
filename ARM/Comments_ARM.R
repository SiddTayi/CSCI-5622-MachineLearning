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


# Load data
setwd("D:/Masters-2023/Machine Learning/ML_Project")
comment_data <- read.csv("all_comments_cleaned.csv")

head(comment_data)

# Comments tab only
df <- comment_data$comments_cleaned_ns
new_df <- data.frame(comments_cleaned_ns = df)


#set.seed(123)
sampled_data = sample_n(new_df, 1234)
# dim(sampled_data)
# head(sampled_data)


# Create an empty csv file to store the tokenized comments
transaction_comments = "all_comment_cleaned_tokenized.csv"

# Open the file
trans <- file(transaction_comments)

# Tokenize the text in the "comments" column
Tokens <- tokenizers::tokenize_words(
  sampled_data$comments_cleaned_ns[1], stopwords = stopwords::stopwords("en"), 
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
Trans <- file(transaction_comments, open = "a")
for (i in 2:nrow(sampled_data)) {
  Tokens <- tokenize_words(sampled_data$comments_cleaned_ns[i],
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



# Reading in the transactional data
comment_df <- read.csv(transaction_comments, header = FALSE, sep = ",")
head(comment_df)

# Convert all columns to char 
comment_df <- comment_df %>%
  mutate_all(as.character)

# (str(comment_df))

# Display the dataframe

comment_df <- comment_df[-c(1), ]
# head(comment_df)


# Read the data to R

CommentTrans = read.transactions("all_comment_cleaned_tokenized.csv",
              
                           rm.duplicates = FALSE, 
                           format = "basket",  
                           sep="," #) 
                           ,cols=1
                           ) 


# inspect(CommentTrans)

# APRIORI
associa_rules <- apriori(data = CommentTrans, 
                        parameter = list(support = 0.0072, 
                                         confidence = 0.01, minlen = 2))

 # Visualising the results
supp <- inspect(sort(associa_rules, by = 'support')[1:15])
conf <- inspect(sort(associa_rules, by = 'confidence')[1:15])
lift <- inspect(sort(associa_rules, by = 'lift')[1:15])

# Plot
itemFrequencyPlot(CommentTrans, topN = 10,
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

plot(associa_rules, 
     method = "graph", 
     measure = "lift", )


# ARM 2

# Use apriori to get the RULES
FrulesK = arules::apriori(CommentTrans, parameter = list(support = 0.0072, 
                                         confidence = 0.014, minlen = 2, maxlen=4))

itemFrequencyPlot(CommentTrans, topN=25, type="absolute")

## Sort rules by a measure such as conf, sup, or lift
SortedRulesK <- sort(FrulesK, by="lift", decreasing=FALSE)
inspect(SortedRulesK[1:15])
(summary(SortedRulesK))

## Sort rules by a measure such as conf, sup, or lift
SortedRulesK <- sort(FrulesK, by="support", decreasing=FALSE)
inspect(SortedRulesK[1:15])
(summary(SortedRulesK))

subrulesK <- head(sort(SortedRulesK, by="lift"),30)
plot(subrulesK)


# 3d plot
plot(subrulesK, method="graph", engine="interactive")