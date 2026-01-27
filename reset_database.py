#!/usr/bin/env python3
"""
Script để reset database - xóa và tạo lại với schema mới
"""

import mysql.connector
import os
import sys

def reset_database():
    """Xóa và tạo lại database"""
    try:
        # Kết nối MySQL
        print("🔌 Kết nối MySQL...")
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            autocommit=True,
            connection_timeout=5
        )
        
        cursor = conn.cursor()
        
        # Drop database nếu tồn tại
        print("🗑️ Xóa database cũ (nếu tồn tại)...")
        try:
            cursor.execute("DROP DATABASE IF EXISTS xla_vantay")
            print("✅ Database cũ đã được xóa")
        except Exception as e:
            print(f"⚠️ Lỗi xóa database: {e}")
        
        # Đọc schema từ file
        schema_file = 'database/schema.sql'
        if not os.path.exists(schema_file):
            print(f"❌ Không tìm thấy {schema_file}")
            return False
        
        print("📖 Đọc schema từ file...")
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Tách các câu lệnh SQL
        statements = schema_sql.split(';')
        
        print("🔨 Tạo database mới...")
        count = 0
        for i, statement in enumerate(statements):
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                    count += 1
                    if i % 5 == 0:
                        print(f"  ✓ Thực hiện {count} câu lệnh...")
                except Exception as e:
                    print(f"⚠️ Lỗi câu lệnh {i}: {e}")
        
        print("✅ Database đã được reset thành công!")
        print("✅ Bây giờ bạn có thể đăng ký người dùng mới với tất cả features")
        
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Lỗi MySQL: {err}")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
        except:
            pass

if __name__ == '__main__':
    success = reset_database()
    sys.exit(0 if success else 1)

