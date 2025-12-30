#!/usr/bin/env python3
"""
SHACL 负对照测试脚本 v2.0
==========================
测试分层验证策略：
- SHACL: 负责配置完整性（结构约束）
- Fast-Check: 负责语义兼容性（运行时规则）

运行方法:
    python test_shacl_negative_control.py

预期结果:
    - 完整配置应该通过 SHACL (conforms=True)
    - 缺少组件的配置应该失败 (conforms=False)
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
    
    # SHACL shapes 路径
    shapes_path = 'shapes/min_shapes.ttl'
    if not Path(shapes_path).exists():
        print(f"❌ SHACL shapes 文件不存在: {shapes_path}")
        return False
    
    print(f"✅ SHACL shapes: {shapes_path}")
    
    # =========================================================================
    # 完整配置（应该通过 SHACL 完整性检查）
    # =========================================================================
    complete_configs = [
        {
            'name': 'Complete_MMS_Cloud',
            'sensor': 'MMS_Riegl_VMX2HA',
            'algorithm': 'Traditional_EdgeDetection',
            'deployment': 'Cloud_AWS_Standard',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
            'data_rate': 50,
        },
        {
            'name': 'Complete_IoT_Edge',
            'sensor': 'IoT_LoRaWAN_Sensor',
            'algorithm': 'Traditional_Threshold',
            'deployment': 'Edge_Local_Server',
            'storage': 'Edge_NAS_Storage',
            'communication': 'Cellular_LTE_Network',
            'inspection_cycle': 7,
            'data_rate': 10,
        },
        {
            'name': 'Complete_UAV_Cloud',
            'sensor': 'UAV_DJI_L1_LiDAR',
            'algorithm': 'ML_RandomForest_Crack',
            'deployment': 'Cloud_Azure_GPU',
            'storage': 'Cloud_Azure_Blob',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 90,
            'data_rate': 100,
        },
        {
            'name': 'Complete_DL_GPU',
            'sensor': 'MMS_Leica_TRK300',
            'algorithm': 'DL_YOLOv8_Crack',
            'deployment': 'Cloud_AWS_GPU',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Fiber_Dedicated_Line',
            'inspection_cycle': 60,
            'data_rate': 80,
        },
    ]
    
    # =========================================================================
    # 不完整配置（应该被 SHACL 拒绝 - 缺少必要组件）
    # =========================================================================
    incomplete_configs = [
        {
            'name': 'Missing_Sensor',
            'description': '缺少传感器',
            'algorithm': 'Traditional_EdgeDetection',
            'deployment': 'Cloud_AWS_Standard',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
        },
        {
            'name': 'Missing_Algorithm',
            'description': '缺少算法',
            'sensor': 'MMS_Riegl_VMX2HA',
            'deployment': 'Cloud_AWS_Standard',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
        },
        {
            'name': 'Missing_Deployment',
            'description': '缺少部署',
            'sensor': 'MMS_Riegl_VMX2HA',
            'algorithm': 'Traditional_EdgeDetection',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
        },
        {
            'name': 'Missing_Storage',
            'description': '缺少存储',
            'sensor': 'MMS_Riegl_VMX2HA',
            'algorithm': 'Traditional_EdgeDetection',
            'deployment': 'Cloud_AWS_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
        },
        {
            'name': 'Missing_Communication',
            'description': '缺少通信',
            'sensor': 'MMS_Riegl_VMX2HA',
            'algorithm': 'Traditional_EdgeDetection',
            'deployment': 'Cloud_AWS_Standard',
            'storage': 'Cloud_S3_Standard',
            'inspection_cycle': 30,
        },
        {
            'name': 'Missing_Multiple',
            'description': '缺少多个组件（sensor, algorithm, deployment）',
            'storage': 'Cloud_S3_Standard',
            'communication': 'Cellular_5G_Network',
            'inspection_cycle': 30,
        },
    ]
    
    # =========================================================================
    # 运行测试
    # =========================================================================
    print("\n" + "=" * 60)
    print("SHACL 完整性验证测试 (v2.0)")
    print("=" * 60)
    print("\n📋 分层验证策略：")
    print("   - SHACL: 检查配置完整性（结构约束）")
    print("   - Fast-Check: 检查语义兼容性（运行时规则）")
    
    results = {
        'complete_tests': [],
        'incomplete_tests': [],
        'summary': {}
    }
    
    # 测试完整配置
    print("\n[1/2] 测试完整配置 (应该通过 SHACL)...")
    complete_pass = 0
    for cfg in complete_configs:
        name = cfg.pop('name', 'Unknown')
        conforms, report = ontology.shacl_validate_config(cfg, shapes_path)
        status = "✅ PASS" if conforms else "❌ FAIL"
        print(f"  {status} {name}")
        
        results['complete_tests'].append({
            'name': name,
            'expected': True,
            'actual': conforms,
            'correct': conforms == True
        })
        
        if conforms:
            complete_pass += 1
        else:
            lines = [l.strip() for l in report.split('\n') if 'Message:' in l or 'Violation' in l]
            for line in lines[:2]:
                print(f"      {line}")
    
    # 测试不完整配置
    print("\n[2/2] 测试不完整配置 (应该被 SHACL 拒绝)...")
    incomplete_fail = 0
    for cfg in incomplete_configs:
        name = cfg.pop('name', 'Unknown')
        desc = cfg.pop('description', '')
        conforms, report = ontology.shacl_validate_config(cfg, shapes_path)
        
        is_correct = not conforms
        status = "✅ CORRECTLY REJECTED" if is_correct else "❌ WRONGLY ACCEPTED"
        print(f"  {status} {name}")
        print(f"      {desc}")
        
        results['incomplete_tests'].append({
            'name': name,
            'description': desc,
            'expected': False,
            'actual': conforms,
            'correct': is_correct
        })
        
        if not conforms:
            incomplete_fail += 1
            lines = [l.strip() for l in report.split('\n') if 'Message:' in l]
            for line in lines[:2]:
                print(f"      {line}")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    total_complete = len(complete_configs)
    total_incomplete = len(incomplete_configs)
    
    print(f"\n完整配置: {complete_pass}/{total_complete} 通过验证")
    print(f"不完整配置: {incomplete_fail}/{total_incomplete} 被正确拒绝")
    
    results['summary'] = {
        'complete_pass_rate': complete_pass / max(1, total_complete),
        'incomplete_reject_rate': incomplete_fail / max(1, total_incomplete),
        'shacl_effective': incomplete_fail > 0
    }
    
    all_complete_pass = complete_pass == total_complete
    all_incomplete_reject = incomplete_fail == total_incomplete
    
    if all_complete_pass and all_incomplete_reject:
        print("\n✅ SHACL 完整性验证器正常工作！")
        print("   - 所有完整配置通过验证")
        print("   - 所有不完整配置被正确拒绝")
        shacl_works = True
    elif not all_complete_pass:
        print("\n⚠️ 警告: 部分完整配置未能通过验证")
        print("   请检查配置图构建逻辑")
        shacl_works = False
    elif not all_incomplete_reject:
        print("\n⚠️ 警告: 部分不完整配置未被拒绝")
        print("   请检查 SHACL shapes 定义")
        shacl_works = False
    else:
        print("\n⚠️ 部分测试未通过")
        shacl_works = False
    
    with open('shacl_negative_control_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 详细结果已保存: shacl_negative_control_results.json")
    
    print("\n" + "-" * 60)
    print("📌 注意：语义兼容性检查（如 GPU↔部署、IoT↔通信）")
    print("   由 evaluation.py 中的 _semantic_fast_check() 方法负责。")
    print("   这是分层验证策略的一部分。")
    print("-" * 60)
    
    return shacl_works


if __name__ == '__main__':
    success = test_shacl_validation()
    sys.exit(0 if success else 1)
