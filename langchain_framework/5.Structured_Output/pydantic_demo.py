from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Harsh"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value representing the cgpa of the student.")


# new_student = {"name":"Harshil"}
# new_student = {"age":"45", "email":"harshil@gmail.com", "cgpa":5}
new_student = {"age":"45", "email":"harshil@gmail.com"}

student = Student(**new_student)

# student_dict = dict(student)
# print(student_dict)


student_json = student.model_dump_json()
print(student_json)

# print(student)
# print(student.name)
# print(type(student.age))
# print(type(student))