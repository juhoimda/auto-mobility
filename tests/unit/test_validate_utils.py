import unittest
import tempfile
import sqlite3
from pathlib import Path
from auto_mobility.utils.validate import validate_db, validate_pointcloud, validate_mesh

class TestValidateUtilsUnit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_db_non_existent(self):
        result = validate_db(str(self.temp_path / "non_existent.db"))
        self.assertFalse(result)

    def test_validate_db_dummy_sqlite(self):
        db_path = self.temp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE NodeData (id INTEGER PRIMARY KEY);")
        cursor.execute("INSERT INTO NodeData VALUES (1);")
        conn.commit()
        conn.close()

        # Dummy size will be small (< 100KB), so validate_db should return False as expected by threshold
        result = validate_db(str(db_path))
        self.assertFalse(result)

    def test_validate_pointcloud_non_existent(self):
        result = validate_pointcloud(str(self.temp_path / "non_existent.ply"))
        self.assertFalse(result)

    def test_validate_mesh_non_existent(self):
        result = validate_mesh(str(self.temp_path / "non_existent.obj"))
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
