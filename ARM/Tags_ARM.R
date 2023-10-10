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


setwd("D:/Masters-2023/Machine Learning/ML_Project")
data = read.csv("Data/Cleaned_video.csv")
head(data, n = 2)

# Comments tab only
df <- data$tags
# head(df)
new_df <- data.frame(tags = df)
# head(new_df)

# Remove square brackets and split the strings into separate elements
new_df$tags <- gsub("\\[|\\]", "", new_df$tags)
new_df$tags <- strsplit(new_df$tags, ", ")

# Flatten the lists and convert to a character vector
new_df$tags <- sapply(new_df$tags, function(x) paste(x, collapse = ", "))

# head(new_df$tags)

# Removing " "
# Remove double quotes from each row in the 'tags' column
new_df$tags <- gsub("\"", "", new_df$tags)
# head(new_df, n = 2)
# dim(new_df)

# Remove special characters and numbers from all columns
new_df[] <- lapply(new_df, function(x) gsub("[^A-Za-z ]", "", x))

sampled_data = sample_n(new_df, 1000)
# Split the string into a vector of items
items <- unlist(strsplit(sampled_data$tags, ", "))
# Create a list of transactions
transactions <- list(items)

transaction_data <- as(transactions, "transactions")
# Inspect the transaction data
inspect(transaction_data)



# Create an empty csv file to store the tokenized comments
tags = "tags.csv"

# Open the file
trans <- file(tags)

# Tokenize the text in the "tags" column
Tokens <- tokenizers::tokenize_words(
  sampled_data$tags[1], stopwords = stopwords::stopwords("en"), 
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
Trans <- file(tags, open = "a")
for (i in 2:nrow(sampled_data)) {
  Tokens <- tokenize_words(sampled_data$tags[i],
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


# Reading the transactional file

tag_df <- read.csv(tags, header = FALSE, sep = ",")
head(comment_df)

# Convert all columns to char 
tag_df <- tag_df %>%
  mutate_all(as.character)

(str(tag_df))

# See the dataframe

tag_df <- tag_df[-c(1), ]
head(tag_df)


TagsTrans = read.transactions("tags.csv",
                            #"KumarGroceriesAS_Transactions.csv",
                           rm.duplicates = FALSE, 
                           format = "basket",  ##if you use "single" also use cols=c(1,2)
                           sep="," #) #,  ## csv file
                           ,cols=1
                           ) ## The dataset HAS row numbers


#inspect(CommentTrans)

#set.seed = 123 # Setting seed
associa_rules <- apriori(data = TagsTrans, 
                        parameter = list(support = 0.01, 
                                         confidence = 0.2, minlen = 2, maxlen = 4))


# Visualising the results
supp <- inspect(sort(associa_rules, by = 'support')[1:15])
conf <- inspect(sort(associa_rules, by = 'confidence')[1:15])
lift <- inspect(sort(associa_rules, by = 'lift')[1:15])

itemFrequencyPlot(TagsTrans, topN = 10,
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
FrulesK = arules::apriori(TagsTrans, parameter = list(support = 0.013, 
                                         confidence = 0.2, minlen = 2, maxlen=4))


# Visualising the results
supp <- inspect(sort(FrulesK, by = 'support')[1:15])
conf <- inspect(sort(FrulesK, by = 'confidence')[1:15])
lift <- inspect(sort(FrulesK, by = 'lift')[1:15])


itemFrequencyPlot(TagsTrans, topN=25, type="absolute")

## Sort rules by a measure such as conf, sup, or lift
SortedRulesK <- sort(FrulesK, by="lift", decreasing=FALSE)
inspect(SortedRulesK[1:5])
(summary(SortedRulesK))

subrulesK <- head(sort(SortedRulesK, by="lift"),30)
plot(subrulesK)

plot(subrulesK, method="graph", engine="interactive")