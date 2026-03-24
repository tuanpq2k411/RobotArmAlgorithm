import math
import numpy as np

L1 = 120
L2 = 120
L3 = 90

def forward_kinematics(a1, a2, a3):
    # Angles in radians
    phi1 = a1
    phi2 = a1 + a2
    phi3 = a1 + a2 + a3
    
    x = L1 * math.cos(phi1) + L2 * math.cos(phi2) + L3 * math.cos(phi3)
    y = L1 * math.sin(phi1) + L2 * math.sin(phi2) + L3 * math.sin(phi3)
    return x, y

def find_angle_gd(target, learning_rate=0.00001, iterations=10000):
    x_d, y_d = target
    
    # Initial guess (radians)
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    
    for i in range(iterations):
        phi1 = a1
        phi2 = a1 + a2
        phi3 = a1 + a2 + a3
        
        x = L1 * math.cos(phi1) + L2 * math.cos(phi2) + L3 * math.cos(phi3)
        y = L1 * math.sin(phi1) + L2 * math.sin(phi2) + L3 * math.sin(phi3)
        
        error_x = x - x_d
        error_y = y - y_d
        
        loss = 0.5 * (error_x**2 + error_y**2)
        
        if loss < 0.1: # Threshold in mm^2
            break
            
        # Gradients
        dx_da1 = -L1 * math.sin(phi1) - L2 * math.sin(phi2) - L3 * math.sin(phi3)
        dx_da2 = -L2 * math.sin(phi2) - L3 * math.sin(phi3)
        dx_da3 = -L3 * math.sin(phi3)
        
        dy_da1 = L1 * math.cos(phi1) + L2 * math.cos(phi2) + L3 * math.cos(phi3)
        dy_da2 = L2 * math.cos(phi2) + L3 * math.cos(phi3)
        dy_da3 = L3 * math.cos(phi3)
        
        de_da1 = error_x * dx_da1 + error_y * dy_da1
        de_da2 = error_x * dx_da2 + error_y * dy_da2
        de_da3 = error_x * dx_da3 + error_y * dy_da3
        
        # Update
        a1 -= learning_rate * de_da1
        a2 -= learning_rate * de_da2
        a3 -= learning_rate * de_da3
        
    return a1, a2, a3

def test():
    targets = [(200, 100), (150, 50), (100, 200), (300, 0)]
    for target in targets:
        a1, a2, a3 = find_angle_gd(target)
        x_res, y_res = forward_kinematics(a1, a2, a3)
        print(f"Target: {target} -> Result: ({x_res:.2f}, {y_res:.2f})")
        print(f"Angles: a1={math.degrees(a1):.2f}, a2={math.degrees(a2):.2f}, a3={math.degrees(a3):.2f}")
        print(f"Servo Angles: A1={math.degrees(a1)+90:.2f}, A2={math.degrees(a2)+90:.2f}, A3={math.degrees(a3)+90:.2f}")
        print("-" * 20)

if __name__ == "__main__":
    test()
