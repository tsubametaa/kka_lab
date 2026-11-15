"""Tests for app.features.ocr.pdf_renderer (upload file validation)."""

import unittest

from app.features.ocr import DocumentValidationError, validate_upload_metadata


class OcrUploadValidationTests(unittest.TestCase):
    def test_validate_upload_metadata_rejects_invalid_extension_empty_and_oversized(self):
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('script.exe', 100)
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('kosong.pdf', 0)
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('besar.pdf', 1024, max_bytes=512)


if __name__ == '__main__':
    unittest.main()
