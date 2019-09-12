import re


class Patterns:
    # department is always one or more alphabetic characters,
    DEPT_PATTERN = r'\A[a-zA-Z]+'
    COURSE_PATTERN = r'\d+'
    # any combination of a Department+Course Number followed by Semester+Year.
    # There is always a space after the Course Number and before Semester+Year.
    # Delimiters are '-', ' ', or ':'.
    COURSE_DEPT_PATTERN = r'\A[a-zA-Z]+[_\s,:]?\d+'
    # A Year is either two digits or four digits.
    YEAR_PATTERN = r'\d{2,4}'
    # Semester abbreviations are F (Fall), W (Winter), S (Spring), Su (Summer)
    SEMESTER_PATTERN = r'[FfWwSs][a-zA-Z]*'


class CourseHero:
    @staticmethod
    def map_semester(string):
        map = { 'F':'F', 'S':'S', 'W':'W', 'SU':'Su', "FALL":"F", "SPRING": "S", "SUMMER":"Su", "WINTER":"W"}
        return map[string.upper()]

    @staticmethod
    def map_year(string):
        if(len(string) != 2 and len(string) != 4):
            raise ValueError("Year should be 2 or 4 digits")
        if(len(string) == 2):
            return int('20' + string)
        return int(string)

    @staticmethod
    def parse(pattern, string):
        tokens = re.findall(pattern=pattern, string=string)
        if len(tokens) > 1:
            raise ValueError("More than one year found")
        elif len(tokens) == 0:
            raise ValueError("No pattern found")
        return tokens[0]

    def __init__(self, string):
        course_department = self.parse(Patterns.COURSE_DEPT_PATTERN, string)
        self.department = self.parse(pattern=Patterns.DEPT_PATTERN, string=course_department).upper()
        self.course_number = int(self.parse(pattern=Patterns.COURSE_PATTERN, string=course_department))

        year_semester = string[len(course_department):].strip()
        self.year = self.map_year(self.parse(pattern=Patterns.YEAR_PATTERN, string=year_semester))
        self.semester = self.map_semester(self.parse(pattern=Patterns.SEMESTER_PATTERN, string=year_semester))

    #  import pdb; pdb.set_trace()

    def __repr__(self):
        return 'Course:\n\tDepartment:{}\n\tCourse Number:{}\n\tYear:{}\n\tSemester:{}' \
            .format(self.department, self.course_number, self.year, self.semester)


if __name__ == '__main__':
    CourseHero("CS12 f16")