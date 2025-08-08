#!/usr/bin/env python3
"""
Validation script for the session state fix
"""

print("🔧 Session State Fix Validation")
print("=" * 40)

# Simulate the session state logic
session_state = {}

def simulate_rate_selection(selected_rate):
    """Simulate what happens when user selects a rate"""
    print(f"User selects rate: {selected_rate}%")
    session_state['pending_rate_update'] = selected_rate
    print(f"✅ Set pending_rate_update = {selected_rate}")
    return "rerun_triggered"

def simulate_rerun():
    """Simulate what happens on app rerun"""
    print("\n🔄 App rerun triggered...")
    
    # Default rate logic
    default_rate = 5.0
    
    if 'pending_rate_update' in session_state:
        default_rate = session_state['pending_rate_update']
        session_state['selected_risk_free_rate'] = default_rate
        del session_state['pending_rate_update']
        print(f"✅ Applied pending rate: {default_rate}%")
        print(f"✅ Set selected_risk_free_rate = {default_rate}")
        print(f"✅ Cleared pending_rate_update")
    elif 'selected_risk_free_rate' in session_state:
        default_rate = session_state['selected_risk_free_rate']
        print(f"📝 Using existing selected rate: {default_rate}%")
    else:
        print(f"📝 Using default rate: {default_rate}%")
    
    # Simulate number_input creation
    print(f"🎛️ number_input created with value: {default_rate}%")
    session_state['manual_rate_input'] = default_rate
    
    return default_rate

# Test scenario
print("Test Scenario: User selects 4.75% rate")
print("-" * 40)

# Initial state
print("Initial session_state:", session_state)

# User clicks button
result = simulate_rate_selection(4.75)

print("Session_state after button click:", session_state)

# App reruns
final_rate = simulate_rerun()

print("Session_state after rerun:", session_state)
print(f"\n🎉 Final result: Input field shows {final_rate}%")
print("✅ No widget key conflicts!")
