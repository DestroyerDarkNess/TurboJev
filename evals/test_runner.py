"""Independent checks for benchmark accounting and gold isolation."""
import argparse
import unittest
from run import matched, numeric_request, jev_request, summarize


class EvaluationTests(unittest.TestCase):
    def test_numeric_gold_is_not_sent(self):
        case={"context":"Invoice contains seven items.","type":"integer","question":"How many items?","expected":123456789,
              "followup":{"question":"Is the earlier answer positive?","expected":True}}
        request=numeric_request(case,argparse.Namespace(thinking="false",max_tokens=64,thinking_budget=None))
        self.assertNotIn("expected",str(request))
        self.assertNotIn("123456789",str(request))
        self.assertEqual(request['questions']['is_positive']['depends_on'],['answer'])

    def test_boolean_is_not_an_integer_answer(self):
        self.assertFalse(matched(True,1))
        self.assertFalse(matched(1,True))
        self.assertTrue(matched(7.0,7))

    def test_errors_stay_in_the_accuracy_denominator(self):
        summary=summarize([{"seconds":1,"correct":True},{"seconds":3,"error":"budget exhausted","correct":False}])
        self.assertEqual(summary['accuracy'],.5)
        self.assertEqual(summary['errors'],1)
        self.assertEqual(summary['latency_seconds']['p95'],3)

    def test_noul_rubric_is_preserved_without_gold(self):
        case={"state":"Evidence","expected":"not_in_prompt","question":{"type":"noul","instructions":"Is it allowed?","criteria":{"true":"Every requirement holds","false":"At least one requirement is missing"}}}
        request=jev_request(case)
        self.assertIn("Every requirement holds",request['questions']['answer']['instructions'])
        self.assertNotIn("not_in_prompt",str(request))
        self.assertNotIn("criteria",request['questions']['answer'])


if __name__=="__main__":
    unittest.main()
