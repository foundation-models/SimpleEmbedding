from unittest import TestCase
from coursehero import CourseHero

class TestCoursehero(TestCase):

    def validate(self, string):
        instance = CourseHero(string)
        self.assertEqual(instance.year,2016)
        self.assertEqual(instance.course_number,111)
        self.assertEqual(instance.semester,"F")
        self.assertEqual(instance.department,"CS")


    def test_init(self):
        self.validate("CS_111 2016 Fall")
        self.validate("CS,111 2016 Fall")
        self.validate("CS 111 F2016")
        self.validate("CS111 2016 Fall")
        self.validate("CS:111 16 Fall")
        self.validate("CS:111 16 fall")
        self.validate("CS:111 f16")

        with self.assertRaises(ValueError):
            CourseHero("CS:111 f203")

        with self.assertRaises(KeyError):
            CourseHero("CS:111 2013 Falll")
