
import math

def calculate_circle_points():
    """
    Calculates and prints the coordinates of 17 points for a closed circular path
    with a diameter of 18.25 meters. The path starts and ends at (0,0,0).
    The output coordinates are in centimeters.
    """
    num_circle_points = 16  # 16 segments for the circle
    diameter_meters = 18.25
    radius_meters = diameter_meters / 2
    radius_cm = radius_meters * 100

    # Offset the circle's center so that the first point lands on the origin (0,0,0)
    center_x = radius_cm
    center_y = 0

    angle_step = (2 * math.pi) / num_circle_points

    points = []
    # Calculate the 16 unique points that form the circle
    for i in range(num_circle_points):
        # Start angle is PI so the first point is at the origin
        angle_rad = math.pi + i * angle_step
        
        x = center_x + radius_cm * math.cos(angle_rad)
        y = center_y + radius_cm * math.sin(angle_rad)
        points.append((x, y, 0.0))

    print("Coordenadas de los 17 puntos (en centímetros):")
    print("---------------------------------------------")
    # Print points 0 to 15
    for i in range(len(points)):
        x, y, z = points[i]
        print(f"Punto {i:>2}: (X={x:8.2f}, Y={y:8.2f}, Z={z:.2f})")
    
    # Print point 16, which is the same as point 0 to close the loop
    x, y, z = points[0]
    print(f"Punto 16: (X={x:8.2f}, Y={y:8.2f}, Z={z:.2f})")

if __name__ == "__main__":
    calculate_circle_points()
