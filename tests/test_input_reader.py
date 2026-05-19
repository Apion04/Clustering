from openpyxl import Workbook
import polars as pl

from src.input_reader import read_supplier_file


def test_read_supplier_xlsx(tmp_path):
    path = tmp_path / "suppliers.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["Supplier Name", "Address"])
    ws.append(["ABC Ltd", "100 King St"])
    wb.save(path)

    df = read_supplier_file(str(path), str(path))

    assert df.columns == ["Supplier Name", "Address"]
    assert df["Supplier Name"].to_list() == ["ABC Ltd"]


def test_read_supplier_csv_preserves_text_values(tmp_path):
    path = tmp_path / "suppliers.csv"
    path.write_text("Name,PurchasingBlocked,Postal\nABC GmbH,FALSE,00123\n", encoding="utf-8")

    df = read_supplier_file(str(path), str(path))

    assert df.schema["PurchasingBlocked"] == pl.String
    assert df["PurchasingBlocked"].to_list() == ["FALSE"]
    assert df["Postal"].to_list() == ["00123"]
