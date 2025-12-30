#!/usr/bin/env python3
"""
SHACL 负对照测试脚本
=====================
证明SHACL validator真的在工作，而不是"永远通过"

运行方法:
    python test_shacl_negative_control.py

预期结果:
    - 5个正常配置应该通过 (conforms=True)
    - 5个故意违规配置应该失败 (conforms=False)
"""

import sys
import json
from pathlib import Path


def test_shacl_validation():
    """测试SHACL验证器的有效性"""

    # 尝试导入
    try:
        from ontology_manager import OntologyManager
    except ImportError:
        print("❌ 无法导入 ontology_manager，请确保在正确目录运行")
        return False

    try:
        import pyshacl
        print("✅ pyshacl 已安装")
    except ImportError:
        print("❌ pyshacl 未安装，请运行: pip install pyshacl")
        return False

    # 初始化本体管理器
    print("\n初始化本体管理器...")
    ontology = OntologyManager()

    # 尝试加载数据
    try:
        ontology.populate_from_csv_files(
            sensor_csv='sensors_data.txt' if Path('sensors_data.txt').exists() else None,
            algorithm_csv='algorithms_data.txt' if Path('algorithms_data.txt').exists() else None,
            infrastructure_csv='infrastructure_data.txt' if Path('infrastructure_data.txt').exists() else None,
        )
    except Exception as e:
        print(f"⚠️ 数据加载警告: {e}")

    # SHACL shapes 路径
    shapes_path = 'shapes/min_shapes.ttl'
    if not Path(shapes_path).exists():
        print(f"❌ SHACL shapes 文件不存在: {shapes_path}")
        print("   请确保已创建 shapes/min_shapes.ttl")
        return False

    print(f"✅ SHACL shapes: {shapes_path}")

    # =========================================================================
    # 正常配置（应该通过）
    # =========================================================================
    valid_configs = [
        {
            'name': 'Valid_MMS_Cloud',
            'sensor': 'MMS_Riegl_VMX2HA',
            'algorithm': 'Traditional_EdgeDetection',
            'deployment': 'Cloud_AWS_Standard',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
            'data_rate': 50,
        },
        {
            'name': 'Valid_IoT_Cellular',
            'sensor': 'IoT_LoRaWAN_Sensor',
            'algorithm': 'Traditional_Threshold',
            'deployment': 'Edge_Local_Server',
            'storage': 'Edge_NAS_Storage',
            'communication': 'Cellular_LTE_Network',
            'inspection_cycle': 7,
            'data_rate': 10,
        },
        {
            'name': 'Valid_UAV_5G',
            'sensor': 'UAV_DJI_L1_LiDAR',
            'algorithm': 'ML_RandomForest_Crack',
            'deployment': 'Cloud_Azure_GPU',
            'storage': 'Cloud_Azure_Blob',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 90,
            'data_rate': 100,
        },
        {
            'name': 'Valid_DL_Cloud',
            'sensor': 'MMS_Leica_TRK300',
            'algorithm': 'DL_YOLOv8_Crack',
            'deployment': 'Cloud_AWS_GPU',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Fiber_Dedicated_Line',
            'inspection_cycle': 60,
            'data_rate': 80,
        },
        {
            'name': 'Valid_Vehicle_Cellular',
            'sensor': 'Vehicle_Smartphone_Camera',
            'algorithm': 'Traditional_ImageAnalysis',
            'deployment': 'Cloud_GCP_Standard',
            'storage': 'Cloud_GCS_Standard',
            'communication': 'Cellular_LTE_Network',
            'inspection_cycle': 14,
            'data_rate': 30,
        },
    ]

    # =========================================================================
    # 故意违规配置（应该失败）
    # =========================================================================
    invalid_configs = [
        {
            'name': 'Invalid_IoT_V2X',
            'description': '规则1违反: IoT固定传感器 + V2X车载通信',
            'sensor': 'IoT_LoRaWAN_Sensor',
            'algorithm': 'Traditional_Threshold',
            'deployment': 'Edge_Local_Server',
            'storage': 'Edge_NAS_Storage',
            'communication': 'V2X_DSRC_Unit',  # 违规：IoT不应用V2X
            'inspection_cycle': 7,
        },
        {
            'name': 'Invalid_DL_OnPremise_NoGPU',
            'description': '规则2违反: DL算法 + OnPremise无GPU',
            'sensor': 'MMS_Leica_TRK300',
            'algorithm': 'DL_YOLOv8_Crack',  # 需要GPU
            'deployment': 'OnPremise_Basic_Server',  # 违规：无GPU
            'storage': 'OnPremise_NAS',
            'communication': 'Fiber_Dedicated_Line',
            'inspection_cycle': 30,
        },
        {
            'name': 'Invalid_Mobile_FiberOnly',
            'description': '规则3违反: 移动传感器 + 仅光纤通信',
            'sensor': 'UAV_DJI_L1_LiDAR',  # 移动传感器
            'algorithm': 'Traditional_PointCloud',
            'deployment': 'Cloud_AWS_Standard',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Fiber_Dedicated_Line',  # 违规：UAV需要无线
            'inspection_cycle': 90,
        },
        {
            'name': 'Invalid_FOS_V2X',
            'description': '规则1违反: 光纤传感器 + V2X通信',
            'sensor': 'FOS_Luna_ODiSI',  # 固定传感器
            'algorithm': 'Traditional_StrainAnalysis',
            'deployment': 'Edge_Local_Server',
            'storage': 'Edge_NAS_Storage',
            'communication': 'V2X_C_V2X_Module',  # 违规
            'inspection_cycle': 1,
        },
        {
            'name': 'Invalid_Missing_Components',
            'description': '完整性违反: 缺少必要组件',
            'sensor': 'MMS_Riegl_VMX2HA',
            # 缺少 algorithm, deployment, storage, communication
            'inspection_cycle': 30,
        },
    ]

    # =========================================================================
    # 运行测试
    # =========================================================================
    print("\n" + "=" * 60)
    print("SHACL 负对照测试")
    print("=" * 60)

    results = {
        'valid_tests': [],
        'invalid_tests': [],
        'summary': {}
    }

    # 测试正常配置
    print("\n[1/2] 测试正常配置 (应该通过)...")
    valid_pass = 0
    for cfg in valid_configs:
        name = cfg.pop('name', 'Unknown')
        conforms, report = ontology.shacl_validate_config(cfg, shapes_path)
        status = "✅ PASS" if conforms else "❌ FAIL"
        print(f"  {status} {name}")

        results['valid_tests'].append({
            'name': name,
            'expected': True,
            'actual': conforms,
            'correct': conforms == True
        })

        if conforms:
            valid_pass += 1
        else:
            print(f"      报告: {report[:200]}...")

    # 测试违规配置
    print("\n[2/2] 测试违规配置 (应该失败)...")
    invalid_fail = 0
    for cfg in invalid_configs:
        name = cfg.pop('name', 'Unknown')
        desc = cfg.pop('description', '')
        conforms, report = ontology.shacl_validate_config(cfg, shapes_path)

        # 对于违规配置，conforms=False才是正确的
        is_correct = not conforms
        status = "✅ CORRECTLY REJECTED" if is_correct else "❌ WRONGLY ACCEPTED"
        print(f"  {status} {name}")
        print(f"      {desc}")

        results['invalid_tests'].append({
            'name': name,
            'description': desc,
            'expected': False,
            'actual': conforms,
            'correct': is_correct
        })

        if not conforms:
            invalid_fail += 1
            # 显示违规原因
            if 'Violation' in report or 'violation' in report:
                lines = [l for l in report.split('\n') if 'Violation' in l or 'Message' in l]
                for line in lines[:3]:
                    print(f"      {line.strip()}")

    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    total_valid = len(valid_configs)
    total_invalid = len(invalid_configs)

    print(f"\n正常配置: {valid_pass}/{total_valid} 通过验证")
    print(f"违规配置: {invalid_fail}/{total_invalid} 被正确拒绝")

    results['summary'] = {
        'valid_pass_rate': valid_pass / total_valid,
        'invalid_reject_rate': invalid_fail / total_invalid,
        'shacl_effective': invalid_fail > 0  # 至少要拒绝一些违规配置
    }

    # 关键判断：SHACL是否真的在工作？
    if invalid_fail == 0:
        print("\n⚠️ 警告: SHACL验证器没有拒绝任何违规配置！")
        print("   可能原因:")
        print("   1. shapes文件中的prefix/属性名与数据不匹配")
        print("   2. targetClass未正确命中配置节点")
        print("   3. SPARQL约束的predicate名称不一致")
        shacl_works = False
    elif invalid_fail == total_invalid:
        print("\n✅ SHACL验证器正常工作：所有违规配置都被正确拒绝")
        shacl_works = True
    else:
        print(f"\n⚠️ 部分有效: {invalid_fail}/{total_invalid} 违规配置被拒绝")
        print("   建议检查未拒绝的配置对应的SHACL约束")
        shacl_works = True

    # 保存结果
    with open('shacl_negative_control_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 详细结果已保存: shacl_negative_control_results.json")

    return shacl_works


if __name__ == '__main__':
    success = test_shacl_validation()
    sys.exit(0 if success else 1)