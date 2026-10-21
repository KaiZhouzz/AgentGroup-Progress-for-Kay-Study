"""沙箱执行器独立测试：读样例 xlsx 算欠费率，应返回 stdout 与 success=True。"""
from app.sandbox.executor import SubprocessExecutor, DockerExecutor, ExecutionResult

CODE = '''from data_files import FILES, OUT_DIR
import pandas as pd

df = pd.read_excel(FILES[0])
total = df["应收"].sum()
owed = df["欠费"].sum()
print(f"总行数: {len(df)}")
print(f"应收总额: {total:.2f}")
print(f"欠费总额: {owed:.2f}")
print(f"欠费率: {owed / total * 100:.2f}%")
'''

if __name__ == "__main__":
    ex = SubprocessExecutor()
    r = ex.execute(CODE, files=["data/samples/billings_2025H1.xlsx"], timeout=60)
    print("success:", r.success)
    print("stdout:\n" + r.stdout)
    if r.stderr:
        print("stderr:\n" + r.stderr)
    if r.error:
        print("error:", r.error)

    bad = ex.execute("import os\nprint(os.listdir('.'))")
    print("forbidden-import blocked:", not bad.success, "|", bad.error)

    try:
        DockerExecutor().execute(CODE)
    except NotImplementedError as e:
        print("docker placeholder raises NotImplementedError:", e)
