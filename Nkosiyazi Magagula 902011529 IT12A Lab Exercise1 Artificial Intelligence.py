# Part 1: 

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Cannot divide by zero"
    return a / b


a = float(input("Enter the first number: "))
b = float(input("Enter the second number: "))

print("Addition:", add(a, b))
print("Subtraction:", subtract(a, b))
print("Multiplication:", multiply(a, b))
print("Division:", divide(a, b))
print("Division by zero example:", divide(a, 0))

# Part 2: 

num_list = [8, 7, 2, 10, 5]

def add_to_list(num):
    num_list.append(num)
    print("List after adding:", num_list)

def remove_from_list(num):
    if num in num_list:
        num_list.remove(num)
    print("List after removing:", num_list)

def average_of_list():
    return sum(num_list) / len(num_list)

def max_in_list():
    return max(num_list)

def min_in_list():
    return min(num_list)

#

add_to_list(4)
remove_from_list(8)
print("Average of list:", average_of_list())
print("Maximum in list:", max_in_list())
print("Minimum in list:", min_in_list())

# Part 3: 

student_scores = {"Anorld": 88, "Boyka": 75, "Charky": 90}

def add_student(name, score):
    student_scores[name] = score
    print("Scores after adding:", student_scores)

def update_score(name, score):
    student_scores[name] = score
    print("Scores after updating:", student_scores)

def remove_student(name):
    if name in student_scores:
        del student_scores[name]
    print("Scores after removing:", student_scores)

def get_score(name):
    return student_scores.get(name, "Student not found")

def class_average():
    return sum(student_scores.values()) / len(student_scores)

def highest_score():
    max_student = max(student_scores, key=student_scores.get)
    return max_student, student_scores[max_student]


add_student("Yazi", 70)
update_score("Anorld", 98)
remove_student("Charky")
print("Score of Boyka:", get_score("Boyka"))
print("Class average score:", class_average())
print("Highest score:", highest_score())