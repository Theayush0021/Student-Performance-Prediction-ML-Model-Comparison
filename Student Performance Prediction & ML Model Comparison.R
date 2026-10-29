library(rpart)
library(ggplot2)

# 1. Load Data
students <- read.csv("StudentsPerformance.csv")

students <- students[complete.cases(
  students[, c("math.score", "reading.score",
               "writing.score", "test.preparation.course")]
), ]

students$test.preparation.course <- factor(
  students$test.preparation.course
)

students$math_result <- factor(
  ifelse(students$math.score >= 40, "Pass", "Fail"),
  levels = c("Fail", "Pass")
)

# 2. EDA
head(students)
tail(students)
str(students)
summary(students)
dim(students)
nrow(students)
ncol(students)

# 3. Train-Test Split
set.seed(123)

index <- sample(
  1:nrow(students),
  floor(0.8 * nrow(students))
)

train_data <- students[index, ]
test_data <- students[-index, ]

# 4. Linear Regression
linear_model <- lm(
  math.score ~ reading.score + writing.score +
    test.preparation.course,
  data = train_data
)

linear_pred <- predict(linear_model, test_data)

# 5. Regression Tree
reg_tree <- rpart(
  math.score ~ reading.score + writing.score +
    test.preparation.course,
  data = train_data,
  method = "anova"
)

tree_reg_pred <- predict(reg_tree, test_data)

# 6. Regression Evaluation
mae_linear <- mean(abs(test_data$math.score - linear_pred))
mae_tree <- mean(abs(test_data$math.score - tree_reg_pred))

cat("Linear Regression MAE:", mae_linear, "\n")
cat("Regression Tree MAE:", mae_tree, "\n")

# 7. Logistic Regression
logistic_model <- glm(
  math_result ~ reading.score + writing.score +
    test.preparation.course,
  data = train_data,
  family = binomial
)

logistic_prob <- predict(
  logistic_model,
  test_data,
  type = "response"
)

logistic_pred <- factor(
  ifelse(logistic_prob >= 0.5, "Pass", "Fail"),
  levels = c("Fail", "Pass")
)

# 8. Classification Tree
class_tree <- rpart(
  math_result ~ reading.score + writing.score +
    test.preparation.course,
  data = train_data,
  method = "class"
)

tree_pred <- predict(
  class_tree,
  test_data,
  type = "class"
)

# 9. Confusion Matrix
cat("Logistic Regression:\n")
print(table(Actual = test_data$math_result,
            Predicted = logistic_pred))

cat("Classification Tree:\n")
print(table(Actual = test_data$math_result,
            Predicted = tree_pred))

# 10. Accuracy
logistic_accuracy <- mean(
  logistic_pred == test_data$math_result
)

tree_accuracy <- mean(
  tree_pred == test_data$math_result
)

cat("Logistic Regression Accuracy:",
    logistic_accuracy, "\n")

cat("Classification Tree Accuracy:",
    tree_accuracy, "\n")

# 11. New Student Prediction
new_student <- data.frame(
  reading.score = 75,
  writing.score = 80,
  test.preparation.course = factor(
    "completed",
    levels = levels(train_data$test.preparation.course)
  )
)

cat("Predicted Math Score:",
    predict(linear_model, new_student), "\n")

cat("Predicted Math Score by Tree:",
    predict(reg_tree, new_student), "\n")

cat("Pass Probability:",
    predict(logistic_model, new_student,
            type = "response"), "\n")

cat("Classification Tree Result:",
    predict(class_tree, new_student,
            type = "class"), "\n")
