import sqlite3
import unittest

from investigation.feature_replay_closeout_v1 import compare_key_cursors


def cursor_for(keys):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE keys (decision_date TEXT, serial_number TEXT)")
    connection.executemany("INSERT INTO keys VALUES (?, ?)", keys)
    connection.commit()
    return connection, connection.execute("SELECT decision_date,serial_number FROM keys ORDER BY decision_date,serial_number")


class CloseoutKeyCompareTest(unittest.TestCase):
    def test_missing_first_middle_last_and_empty_side_terminate_with_diffs(self):
        full = [("2023-01-01", "a"), ("2023-01-02", "b"), ("2023-01-03", "c")]
        cases = [
            full[1:],       # missing first
            [full[0], full[2]],  # missing middle
            full[:2],       # missing last
            [],             # empty right side
        ]
        for other in cases:
            left_conn, left = cursor_for(full)
            right_conn, right = cursor_for(other)
            result = compare_key_cursors(left, right)
            self.assertTrue(result["mismatches_first20"], other)
            left_conn.close()
            right_conn.close()


if __name__ == "__main__":
    unittest.main()

