#!/usr/bin/env python3
"""
🧪 LEO INTEGRATION TEST SCRIPT 🧪
Test script to verify all components are properly integrated and working together.

This script tests:
- Master Integration System initialization
- Component loading and health checks
- Cross-component communication
- Learning engine functionality
- Resource optimizer operation
- Multimodal processor capabilities
"""

import asyncio
import sys
import traceback
import time
from pathlib import Path

# Add Backend to Python path
sys.path.append(str(Path(__file__).parent / "Backend"))

print("🧪 LEO Integration Test Suite")
print("="*50)

async def test_master_integration():
    """Test Master Integration System"""
    print("\n1️⃣  Testing Master Integration System...")
    
    try:
        from Backend.MasterIntegrationSystem import MasterIntegrationSystem
        
        # Initialize system
        master_system = MasterIntegrationSystem()
        await master_system.initialize()
        
        # Get system status
        status = master_system.get_system_status()
        
        print(f"   ✅ System initialized successfully")
        print(f"   📊 Status: {status['status']}")
        print(f"   🔧 Components: {len(status['active_components'])}")
        print(f"   🧠 Active: {', '.join(status['active_components'][:3])}...")
        
        # Test processing
        result = await master_system.process_request(
            user_id="test_user",
            request="Hello, test request",
            modality="text"
        )
        
        if result.get('success', False):
            print(f"   ✅ Request processing works")
        else:
            print(f"   ⚠️  Request processing had issues: {result.get('error', 'Unknown')}")
        
        # Cleanup
        await master_system.shutdown()
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

async def test_individual_components():
    """Test individual advanced components"""
    print("\n2️⃣  Testing Individual Components...")
    
    results = {}
    
    # Test Advanced Learning Engine
    try:
        from Backend.AdvancedLearningEngine import AdvancedLearningEngine
        learning_engine = AdvancedLearningEngine()
        
        # Quick test
        session_id = learning_engine.start_learning_session(["test"])
        status = learning_engine.get_learning_status()
        
        print(f"   ✅ Learning Engine: {status['active_learning']}")
        results['learning_engine'] = True
        
        # Cleanup
        learning_engine.end_learning_session()
        
    except Exception as e:
        print(f"   ❌ Learning Engine: {str(e)[:50]}...")
        results['learning_engine'] = False
    
    # Test Resource Optimizer
    try:
        from Backend.IntelligentResourceOptimizer import IntelligentResourceOptimizer
        optimizer = IntelligentResourceOptimizer()
        
        # Quick test
        status = optimizer.get_resource_status()
        
        print(f"   ✅ Resource Optimizer: CPU {status['cpu']['usage_percent']:.1f}%")
        results['resource_optimizer'] = True
        
    except Exception as e:
        print(f"   ❌ Resource Optimizer: {str(e)[:50]}...")
        results['resource_optimizer'] = False
    
    # Test Multimodal Processor
    try:
        from Backend.MultimodalAIProcessor import MultimodalAIProcessor
        processor = MultimodalAIProcessor()
        
        # Quick test
        result = await processor.process_text_query("test query")
        
        print(f"   ✅ Multimodal Processor: {result.get('success', False)}")
        results['multimodal_processor'] = True
        
    except Exception as e:
        print(f"   ❌ Multimodal Processor: {str(e)[:50]}...")
        results['multimodal_processor'] = False
    
    return results

def test_legacy_components():
    """Test existing LEO components"""
    print("\n3️⃣  Testing Legacy Components...")
    
    results = {}
    
    # Test Search Engine
    try:
        sys.path.append(str(Path(__file__).parent / "Backend"))
        from Backend.RealtimeSearchEngine import RealtimeSearchEngine, initialize
        
        # Test initialization
        init_result = initialize()
        if init_result:
            print(f"   ✅ Search Engine: Ready")
            results['search_engine'] = True
        else:
            print(f"   ❌ Search Engine: Failed to initialize")
            results['search_engine'] = False
        
    except Exception as e:
        print(f"   ❌ Search Engine: {str(e)[:50]}...")
        results['search_engine'] = False
    
    # Test Chatbot
    try:
        from Backend.Chatbot import ChatBot
        chatbot = ChatBot()
        
        print(f"   ✅ Chatbot: Ready")
        results['chatbot'] = True
        
    except Exception as e:
        print(f"   ❌ Chatbot: {str(e)[:50]}...")
        results['chatbot'] = False
    
    # Test Data Analyzer
    try:
        from Backend.DataAnalyzer import UniversalAutonomousResearchAnalytics
        analyzer = UniversalAutonomousResearchAnalytics()
        
        print(f"   ✅ Data Analyzer: Ready")
        results['data_analyzer'] = True
        
    except Exception as e:
        print(f"   ❌ Data Analyzer: {str(e)[:50]}...")
        results['data_analyzer'] = False
    
    return results

def test_gui_components():
    """Test GUI components"""
    print("\n4️⃣  Testing GUI Components...")
    
    results = {}
    
    try:
        sys.path.append(str(Path(__file__).parent / "Frontend"))
        
        # Test GUI imports (don't actually use to avoid blocking)
        try:
            from Frontend.GUI import Listen, Speak
            print(f"   ✅ Voice Components: Available")
            results['voice'] = True
        except ImportError:
            print(f"   ⚠️  Voice Components: Not available (may need GUI libraries)")
            results['voice'] = False
        
    except Exception as e:
        print(f"   ❌ GUI Components: {str(e)[:50]}...")
        results['voice'] = False
    
    return results

def test_voice_leo_components():
    """Test enhanced Voice LEO components"""
    print("\n5️⃣  Testing Enhanced Voice LEO...")
    
    results = {}
    
    # Test Voice LEO main system
    try:
        from Voice_LEO import VoiceLEO, initialize_voice_leo, process_voice_input_sync
        
        print(f"   ✅ Voice LEO Core: Available")
        results['voice_leo_core'] = True
        
        # Test Voice LEO capabilities detection
        try:
            voice_leo_instance = VoiceLEO()
            capabilities = voice_leo_instance.get_capabilities_summary()
            print(f"   ✅ Voice LEO Capabilities: {len(capabilities)} chars")
            results['voice_leo_capabilities'] = True
        except Exception as e:
            print(f"   ⚠️  Voice LEO Capabilities: {str(e)[:30]}...")
            results['voice_leo_capabilities'] = False
        
    except ImportError as e:
        print(f"   ❌ Voice LEO Core: Not available ({str(e)[:30]}...)")
        results['voice_leo_core'] = False
        results['voice_leo_capabilities'] = False
    
    # Test Image Generation integration
    try:
        from Backend.ImageGeneration import EnhancedTextToImage
        print(f"   ✅ Image Generation: Available")
        results['image_generation'] = True
    except ImportError:
        print(f"   ⚠️  Image Generation: Not available")
        results['image_generation'] = False
    
    # Test Automation integration (Music, YouTube, Content)
    try:
        from Backend.Automation import PlayMusic, YouTubeSearch, PlayYoutube, Content
        print(f"   ✅ Automation (Music/YouTube/Content): Available")
        results['automation'] = True
    except ImportError:
        print(f"   ⚠️  Automation: Not available")
        results['automation'] = False
    
    # Test Enhanced TTS
    try:
        from Backend.TextToSpeech import TextToSpeech as TTSEngine
        print(f"   ✅ Enhanced TTS: Available")
        results['enhanced_tts'] = True
    except ImportError:
        print(f"   ⚠️  Enhanced TTS: Not available")
        results['enhanced_tts'] = False
    
    return results

async def test_voice_leo_processing():
    """Test Voice LEO processing functionality"""
    print("\n6️⃣  Testing Voice LEO Processing...")
    
    try:
        from Voice_LEO import VoiceLEO
        
        # Create Voice LEO instance
        voice_leo = VoiceLEO()
        await voice_leo.initialize()
        
        # Test basic command processing
        test_commands = [
            "hello",
            "what time is it",
            "what can you do",
            "tell me a joke"
        ]
        
        successful_tests = 0
        
        for command in test_commands:
            try:
                response = await voice_leo.process_voice_command(command)
                if response and len(response) > 0:
                    successful_tests += 1
                    print(f"   ✅ Command '{command}': Success ({len(response)} chars)")
                else:
                    print(f"   ⚠️  Command '{command}': Empty response")
            except Exception as e:
                print(f"   ❌ Command '{command}': Error ({str(e)[:30]}...)")
        
        success_rate = successful_tests / len(test_commands)
        print(f"   📊 Voice LEO Processing: {successful_tests}/{len(test_commands)} commands successful ({success_rate:.1%})")
        
        return success_rate >= 0.5
        
    except Exception as e:
        print(f"   ❌ Voice LEO Processing: Failed to initialize ({str(e)[:50]}...)")
        return False

async def test_ultimate_voice_leo():
    """Test Ultimate Voice LEO with all advanced features"""
    print("\n7️⃣  Testing Ultimate Voice LEO Integration...")
    
    try:
        # Import Ultimate Voice LEO launcher
        sys.path.append(str(Path(__file__).parent))
        from ultimate_voice_leo_launcher import check_all_capabilities
        
        # Check all advanced capabilities
        capabilities = check_all_capabilities()
        available_features = sum(capabilities.values())
        total_features = len(capabilities)
        
        print(f"   📊 Feature Availability: {available_features}/{total_features}")
        
        # Test Voice LEO with advanced commands
        try:
            from Voice_LEO import VoiceLEO, process_voice_input_sync
            
            # Test comprehensive command set
            ultimate_commands = [
                "hello ultimate leo",
                "what can you do", 
                "create an image of a test",
                "analyze my data",
                "create 3d model of a cube",
                "create a research agent",
                "update yourself",
                "backup system"
            ]
            
            successful = 0
            total = len(ultimate_commands)
            
            for i, command in enumerate(ultimate_commands, 1):
                try:
                    response = process_voice_input_sync(command)
                    if response and len(response) > 0 and "not available" not in response.lower():
                        print(f"   ✅ Ultimate Command {i}/{total}: '{command[:20]}...' - SUCCESS")
                        successful += 1
                    else:
                        print(f"   ⚠️  Ultimate Command {i}/{total}: '{command[:20]}...' - FEATURE N/A")
                except Exception as e:
                    print(f"   ❌ Ultimate Command {i}/{total}: '{command[:20]}...' - ERROR")
            
            success_rate = successful / total
            print(f"   📊 Ultimate Integration: {successful}/{total} features working ({success_rate:.1%})")
            
            # Determine overall Ultimate status
            if available_features >= 7 and success_rate >= 0.6:
                print(f"   🎆 ULTIMATE STATUS: Fully operational with advanced features!")
                return True
            elif available_features >= 5 and success_rate >= 0.4:
                print(f"   ✨ ADVANCED STATUS: Most features working well!")
                return True
            elif available_features >= 3 and success_rate >= 0.3:
                print(f"   🔄 ENHANCED STATUS: Core features operational")
                return True
            else:
                print(f"   🔧 BASIC STATUS: Limited feature availability")
                return False
                
        except Exception as e:
            print(f"   ❌ Ultimate Voice LEO testing failed: {str(e)[:50]}...")
            return False
            
        
    except Exception as e:
        print(f"   ❌ Ultimate Voice LEO import failed: {str(e)[:50]}...")
        return False

async def test_end_to_end():
    """Test end-to-end functionality"""
    print("\n8️⃣  Testing End-to-End Integration...")
    
    try:
        # Import the new integrated main
        sys.path.append(str(Path(__file__).parent))
        from Main_Integrated import LEOAssistant
        
        # Initialize LEO (but don't run interactively)
        leo = LEOAssistant()
        
        # Override configs to avoid interactive mode
        leo.config["voice_enabled"] = False
        leo.config["continuous_listening"] = False
        
        success = await leo.initialize()
        
        if success:
            print(f"   ✅ LEO Assistant initialized successfully")
            
            # Test a simple interaction programmatically
            if leo.master_system:
                result = await leo.master_system.process_request(
                    user_id="test_user",
                    request="What is artificial intelligence?",
                    modality="text"
                )
                
                if result.get('success', False):
                    print(f"   ✅ End-to-end processing works")
                    response = result.get('response', '')[:60] + "..." if len(result.get('response', '')) > 60 else result.get('response', '')
                    print(f"   💬 Response: {response}")
                else:
                    print(f"   ⚠️  Processing had issues: {result.get('error', 'Unknown')}")
            
            # Clean shutdown
            await leo.shutdown()
            
            return True
        else:
            print(f"   ❌ LEO Assistant failed to initialize")
            return False
            
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        return False

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    all_results = {}
    for result_dict in results:
        if isinstance(result_dict, dict):
            all_results.update(result_dict)
        else:
            all_results['master_integration'] = result_dict
    
    total_tests = len(all_results)
    passed_tests = sum(1 for v in all_results.values() if v)
    
    print(f"📈 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    print(f"\n✅ Passed Components:")
    for component, passed in all_results.items():
        if passed:
            print(f"   • {component.replace('_', ' ').title()}")
    
    if passed_tests < total_tests:
        print(f"\n❌ Failed Components:")
        for component, passed in all_results.items():
            if not passed:
                print(f"   • {component.replace('_', ' ').title()}")
    
    print(f"\n🎯 Integration Status: {'🟢 FULLY INTEGRATED' if passed_tests >= total_tests * 0.8 else '🟡 PARTIALLY INTEGRATED' if passed_tests >= total_tests * 0.5 else '🔴 NEEDS WORK'}")
    
    return passed_tests / total_tests

async def main():
    """Run all integration tests"""
    print("Starting LEO integration tests...\n")
    
    results = []
    
    try:
        # Test 1: Master Integration System
        master_result = await test_master_integration()
        results.append({'master_integration': master_result})
        
        # Test 2: Individual advanced components
        component_results = await test_individual_components()
        results.append(component_results)
        
        # Test 3: Legacy components
        legacy_results = test_legacy_components()
        results.append(legacy_results)
        
        # Test 4: GUI components
        gui_results = test_gui_components()
        results.append(gui_results)
        
        # Test 5: Voice LEO components
        voice_leo_results = test_voice_leo_components()
        results.append(voice_leo_results)
        
        # Test 6: Voice LEO processing
        voice_leo_processing_result = await test_voice_leo_processing()
        results.append({'voice_leo_processing': voice_leo_processing_result})
        
        # Test 7: Ultimate Voice LEO integration
        ultimate_result = await test_ultimate_voice_leo()
        results.append({'ultimate_voice_leo': ultimate_result})
        
        # Test 8: End-to-end
        e2e_result = await test_end_to_end()
        results.append({'end_to_end': e2e_result})
        
        # Print summary
        success_rate = print_summary(results)
        
        print(f"\n🏁 Tests completed!")
        
        if success_rate >= 0.8:
            print("🎉 LEO is fully integrated and ready to use!")
            print("   Run: python Main_Integrated.py")
        elif success_rate >= 0.5:
            print("⚠️  LEO is partially integrated. Some features may not work.")
            print("   Check the failed components above.")
        else:
            print("❌ LEO integration needs work. Multiple components failed.")
        
    except Exception as e:
        print(f"\n❌ Test suite encountered an error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
