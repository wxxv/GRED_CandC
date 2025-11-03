import sqlite3
import csv
import os
from pathlib import Path


def convert_sqlite_to_csv(database_folder="database", output_base_folder="database_csv"):
    """
    将 database 文件夹中的所有 SQLite 文件转换为 CSV 文件
    每个 SQLite 文件创建一个文件夹，表转换为该文件夹中的 CSV 文件
    
    Args:
        database_folder: 包含 SQLite 文件的文件夹路径
        output_base_folder: 输出 CSV 文件的基础文件夹路径
    """
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    db_folder_path = script_dir / database_folder
    output_base_path = script_dir / output_base_folder
    
    # 确保数据库文件夹存在
    if not db_folder_path.exists():
        print(f"错误: 数据库文件夹 '{db_folder_path}' 不存在!")
        return
    
    # 创建输出基础文件夹
    output_base_path.mkdir(exist_ok=True)
    
    # 获取所有 SQLite 文件
    sqlite_files = list(db_folder_path.glob("*.sqlite"))
    
    if not sqlite_files:
        print(f"警告: 在 '{db_folder_path}' 中没有找到 SQLite 文件!")
        return
    
    print(f"找到 {len(sqlite_files)} 个 SQLite 文件")
    print("-" * 80)
    
    # 处理每个 SQLite 文件
    for idx, sqlite_file in enumerate(sqlite_files, 1):
        # 获取文件名（不含扩展名）
        db_name = sqlite_file.stem
        print(f"\n[{idx}/{len(sqlite_files)}] 处理: {sqlite_file.name}")
        
        # 为该数据库创建输出文件夹
        output_folder = output_base_path / db_name
        output_folder.mkdir(exist_ok=True)
        
        try:
            # 连接到 SQLite 数据库
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                print(f"  警告: 数据库中没有找到表")
                conn.close()
                continue
            
            print(f"  找到 {len(tables)} 个表")
            
            # 导出每个表为 CSV
            for table_name, in tables:
                try:
                    # 跳过 SQLite 内部表
                    if table_name.startswith('sqlite_'):
                        continue
                    
                    # 查询表中的所有数据
                    cursor.execute(f"SELECT * FROM `{table_name}`")
                    rows = cursor.fetchall()
                    
                    # 获取列名
                    column_names = [description[0] for description in cursor.description]
                    
                    # CSV 文件路径
                    csv_file = output_folder / f"{table_name}.csv"
                    
                    # 写入 CSV 文件（使用 utf-8-sig 编码以避免 Excel 乱码）
                    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        # 写入表头
                        writer.writerow(column_names)
                        # 写入数据行
                        # 处理可能的 NULL 值
                        cleaned_rows = []
                        for row in rows:
                            cleaned_row = [str(cell) if cell is not None else '' for cell in row]
                            cleaned_rows.append(cleaned_row)
                        writer.writerows(cleaned_rows)
                    
                    print(f"    ✓ {table_name}.csv ({len(rows)} 行)")
                    
                except Exception as e:
                    print(f"    ✗ 导出表 '{table_name}' 失败: {e}")
            
            # 关闭数据库连接
            conn.close()
            
        except Exception as e:
            print(f"  错误: 无法处理数据库文件: {e}")
    
    print("\n" + "=" * 80)
    print(f"转换完成! CSV 文件已保存到: {output_base_path}")


if __name__ == "__main__":
    # 执行转换
    convert_sqlite_to_csv()
    
    print("\n提示: 如需修改输入/输出文件夹，可以调用:")
    print("  convert_sqlite_to_csv(database_folder='你的路径', output_base_folder='输出路径')")

