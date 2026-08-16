import array
import io
import struct
import unittest

from tools.convert import HEADER, MAGIC, gpt2_byte_decoder, quantize_value, write_matrix


class FakeSafeTensor:
    def tensor(self, _name):
        return [1, 3], array.array("f", [1.0, -2.0, 0.5])


class ConverterTests(unittest.TestCase):
    def test_header_contract(self):
        self.assertEqual(HEADER.size, 112)
        self.assertEqual(len(MAGIC), 8)

    def test_q8_clamps(self):
        self.assertEqual(quantize_value(2.0, 0.01, 8), 127)
        self.assertEqual(quantize_value(-2.0, 0.01, 8), -127)

    def test_q4_clamps(self):
        self.assertEqual(quantize_value(2.0, 0.1, 4), 7)
        self.assertEqual(quantize_value(-2.0, 0.1, 4), -7)

    def test_q2_is_ternary(self):
        self.assertEqual(quantize_value(-0.6, 1.0, 2), -1)
        self.assertEqual(quantize_value(0.1, 1.0, 2), 0)
        self.assertEqual(quantize_value(0.6, 1.0, 2), 1)

    def test_bytelevel_map_is_bijective(self):
        decoder = gpt2_byte_decoder()
        self.assertEqual(len(decoder), 256)
        self.assertEqual(set(decoder.values()), set(range(256)))

    def test_fp16_matrix_is_raw(self):
        output = io.BytesIO()
        write_matrix(output, FakeSafeTensor(), "matrix", 1, 3, 16)
        self.assertEqual(struct.unpack("<3e", output.getvalue()), (1.0, -2.0, 0.5))

    def test_fp32_matrix_is_raw(self):
        output = io.BytesIO()
        write_matrix(output, FakeSafeTensor(), "matrix", 1, 3, 32)
        self.assertEqual(struct.unpack("<3f", output.getvalue()), (1.0, -2.0, 0.5))

    def test_odd_width_q4_row_has_exact_packed_size(self):
        output = io.BytesIO()
        write_matrix(output, FakeSafeTensor(), "matrix", 1, 3, 4)
        self.assertEqual(len(output.getvalue()), 6)


if __name__ == "__main__":
    unittest.main()
