# COMP0233 Exam Prep — 8-Week Practical Study Guide

**Module:** Research Software Engineering with Python (UCL)
**Exam type:** Practical coding
**Start date:** March 2026 | **Exam:** ~May 2026

---

## How to Use This Guide

Each week has three parts:
1. **Concepts** — what you need to understand
2. **Tutorials** — worked examples to type out and study
3. **Exercises** — problems to solve yourself (solutions at the end of each week)

Work through everything in a Jupyter notebook or `.py` files. **Type the code — don't copy-paste.** The muscle memory matters for a practical exam.

---

# WEEKS 1–2: Testing & CI

## Week 1: pytest Fundamentals

### Concepts
- Unit tests verify small, isolated pieces of code
- Regression tests ensure changes don't break existing behaviour
- `pytest` discovers any function starting with `test_` in files starting with `test_`
- `assert` is the core mechanism — pytest gives rich failure output
- `pytest.approx()` handles floating-point comparison with configurable tolerance
- `pytest.raises()` tests that exceptions are raised correctly (negative testing)

### Tutorial 1.1: Your First Test Suite

Create a file called `geometry.py`:

```python
# geometry.py
import math

def rectangle_area(width, height):
    """Calculate area of a rectangle."""
    if width < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return width * height

def circle_area(radius):
    """Calculate area of a circle."""
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * radius ** 2

def triangle_area(base, height):
    """Calculate area of a triangle."""
    if base < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative")
    return 0.5 * base * height

def overlap_area(rect1, rect2):
    """
    Calculate overlap area of two rectangles.
    Each rect is (x_min, y_min, x_max, y_max).
    Returns 0.0 if no overlap.
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    return x_overlap * y_overlap
```

Now create `test_geometry.py`:

```python
# test_geometry.py
import pytest
from pytest import approx
from geometry import rectangle_area, circle_area, triangle_area, overlap_area
import math

# --- Basic assertions ---

def test_rectangle_area_simple():
    assert rectangle_area(3, 4) == 12

def test_rectangle_area_float():
    assert rectangle_area(2.5, 4.0) == 10.0

def test_rectangle_area_zero():
    assert rectangle_area(0, 5) == 0

# --- Floating point comparison with approx ---

def test_circle_area():
    # pi * 1^2 = pi ≈ 3.14159...
    assert circle_area(1) == approx(math.pi)

def test_circle_area_tolerance():
    # approx default relative tolerance is 1e-6
    assert circle_area(10) == approx(314.159265, rel=1e-5)

# --- Negative testing with pytest.raises ---

def test_rectangle_negative_width():
    with pytest.raises(ValueError):
        rectangle_area(-1, 5)

def test_circle_negative_radius():
    with pytest.raises(ValueError, match="non-negative"):
        circle_area(-3)

# --- Overlap tests (from the course's Saskatchewan example) ---

def test_full_overlap():
    assert overlap_area((1, 1, 4, 4), (2, 2, 3, 3)) == 1.0

def test_partial_overlap():
    assert overlap_area((1, 1, 4, 4), (2, 2, 3, 4.5)) == 2.0

def test_no_overlap():
    assert overlap_area((1, 1, 4, 4), (5, 5, 6, 6)) == 0.0

def test_touching_edges():
    # Rectangles share an edge but no area
    assert overlap_area((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0
```

Run with: `pytest test_geometry.py -v`

### Tutorial 1.2: Parametrize

When you have many similar test cases, `@pytest.mark.parametrize` avoids repetition:

```python
# test_geometry_parametrized.py
import pytest
from geometry import rectangle_area, overlap_area

@pytest.mark.parametrize("width,height,expected", [
    (3, 4, 12),
    (0, 5, 0),
    (2.5, 4.0, 10.0),
    (1, 1, 1),
    (100, 0.01, 1.0),
])
def test_rectangle_area_cases(width, height, expected):
    assert rectangle_area(width, height) == expected

@pytest.mark.parametrize("rect1,rect2,expected", [
    ((1, 1, 4, 4), (2, 2, 3, 3), 1.0),     # full containment
    ((1, 1, 4, 4), (2, 2, 3, 4.5), 2.0),    # partial
    ((1, 1, 4, 4), (5, 5, 6, 6), 0.0),      # no overlap
    ((0, 0, 1, 1), (1, 0, 2, 1), 0.0),      # touching edge
    ((0, 0, 2, 2), (0, 0, 2, 2), 4.0),      # identical
])
def test_overlap_cases(rect1, rect2, expected):
    assert overlap_area(rect1, rect2) == expected
```

### Exercise 1.1: Write Tests for a Statistics Module

Here is a module. **Write a comprehensive test file for it.**

```python
# stats.py
import math

def mean(data):
    if not data:
        raise ValueError("Cannot compute mean of empty list")
    return sum(data) / len(data)

def variance(data):
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)

def std_dev(data):
    return math.sqrt(variance(data))

def median(data):
    if not data:
        raise ValueError("Cannot compute median of empty list")
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
```

Your tests should cover:
- Normal cases with known results
- Edge cases (single element, two elements, even/odd length lists)
- Floating-point results using `approx`
- Exception cases using `pytest.raises`
- Use `@pytest.mark.parametrize` for at least one test function

<details>
<summary><strong>Solution 1.1</strong></summary>

```python
# test_stats.py
import pytest
from pytest import approx
from stats import mean, variance, std_dev, median

# --- mean ---

@pytest.mark.parametrize("data,expected", [
    ([1, 2, 3], 2.0),
    ([10], 10.0),
    ([1.5, 2.5], 2.0),
    ([-1, 0, 1], 0.0),
])
def test_mean(data, expected):
    assert mean(data) == approx(expected)

def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])

# --- variance ---

def test_variance_known():
    # [2, 4, 4, 4, 5, 5, 7, 9] -> variance = 4.571...
    assert variance([2, 4, 4, 4, 5, 5, 7, 9]) == approx(4.571428571, rel=1e-5)

def test_variance_identical():
    assert variance([5, 5, 5, 5]) == 0.0

def test_variance_two_points():
    # [0, 10] -> mean=5, variance = ((25+25)/1) = 50
    assert variance([0, 10]) == approx(50.0)

def test_variance_single_element():
    with pytest.raises(ValueError):
        variance([42])

# --- std_dev ---

def test_std_dev():
    assert std_dev([2, 4, 4, 4, 5, 5, 7, 9]) == approx(2.13809, rel=1e-3)

# --- median ---

@pytest.mark.parametrize("data,expected", [
    ([1, 3, 5], 3),           # odd length
    ([1, 2, 3, 4], 2.5),      # even length
    ([7], 7),                  # single
    ([5, 1, 3], 3),           # unsorted input
    ([1, 2], 1.5),            # two elements
])
def test_median(data, expected):
    assert median(data) == approx(expected)

def test_median_empty():
    with pytest.raises(ValueError):
        median([])
```
</details>

---

## Week 2: Mocking & Fixtures

### Concepts
- **Mocking** replaces real objects/functions with controllable fakes
- Use `unittest.mock.patch` to replace a module's function temporarily
- Use `unittest.mock.MagicMock` for objects whose return values need to support arithmetic
- `mock.assert_called_with(...)` verifies the mock was called with specific arguments
- `mock.assert_any_call(...)` checks if a particular call happened at any point
- **Fixtures** in pytest provide reusable setup/teardown with `@pytest.fixture`

### Tutorial 2.1: Mocking External Calls

Imagine you have a function that fetches data from the internet:

```python
# weather.py
import requests

def get_temperature(city):
    """Fetch current temperature for a city from an API."""
    response = requests.get(
        "https://api.weather.example.com/current",
        params={"city": city, "units": "celsius"}
    )
    data = response.json()
    return data["temperature"]

def is_freezing(city):
    """Check if temperature is below zero."""
    return get_temperature(city) < 0
```

You can't call a real API in tests. Mock it:

```python
# test_weather.py
from unittest.mock import patch, MagicMock
from weather import get_temperature, is_freezing

def test_get_temperature():
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.json.return_value = {"temperature": 21.5}

    with patch("weather.requests.get", return_value=mock_response) as mock_get:
        result = get_temperature("London")

        # Verify the result
        assert result == 21.5

        # Verify the API was called correctly
        mock_get.assert_called_with(
            "https://api.weather.example.com/current",
            params={"city": "London", "units": "celsius"}
        )

def test_is_freezing_true():
    with patch("weather.get_temperature", return_value=-5):
        assert is_freezing("Moscow") is True

def test_is_freezing_false():
    with patch("weather.get_temperature", return_value=25):
        assert is_freezing("Cairo") is False
```

### Tutorial 2.2: Mocking to Test Algorithm Behaviour

From the course — testing a numerical derivative without knowing the function:

```python
# calculus.py
def partial_derivative(function, at, direction, delta=1.0):
    """Compute partial derivative of function at a point."""
    f_x = function(at)
    x_plus_delta = at[:]
    x_plus_delta[direction] += delta
    f_x_plus_delta = function(x_plus_delta)
    return (f_x_plus_delta - f_x) / delta
```

```python
# test_calculus.py
from unittest.mock import MagicMock
from calculus import partial_derivative

def test_derivative_calls_function_correctly():
    """Verify partial_derivative calls function at the right points."""
    func = MagicMock()
    partial_derivative(func, [0, 0], 1)  # derivative in y-direction

    # Should have been called at [0, 0] and [0, 1.0]
    func.assert_any_call([0, 0])
    func.assert_any_call([0, 1.0])

def test_derivative_x_direction():
    func = MagicMock()
    partial_derivative(func, [5, 3], 0, delta=0.5)

    func.assert_any_call([5, 3])
    func.assert_any_call([5.5, 3])

def test_derivative_result():
    """Test with a known function: f(x,y) = x^2 + y^2"""
    def f(point):
        return point[0]**2 + point[1]**2

    # df/dx at (3, 4) = 2*3 = 6
    result = partial_derivative(f, [3, 4], 0, delta=0.0001)
    assert abs(result - 6.0) < 0.01
```

### Tutorial 2.3: pytest Fixtures

```python
# test_with_fixtures.py
import pytest
import os
import json

@pytest.fixture
def sample_data():
    """Provides sample data for tests."""
    return [
        {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 92},
        {"name": "Charlie", "score": 78},
    ]

@pytest.fixture
def temp_json_file(tmp_path, sample_data):
    """Creates a temporary JSON file with sample data."""
    filepath = tmp_path / "data.json"
    filepath.write_text(json.dumps(sample_data))
    return filepath

def test_load_data(temp_json_file):
    with open(temp_json_file) as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[0]["name"] == "Alice"

def test_average_score(sample_data):
    avg = sum(d["score"] for d in sample_data) / len(sample_data)
    assert avg == pytest.approx(85.0)
```

### Exercise 2.1: Mock a Database

Given this code that queries a database:

```python
# user_service.py
class Database:
    def query(self, sql, params=None):
        """Execute SQL query and return results."""
        raise NotImplementedError("Real DB connection needed")

class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        results = self.db.query(
            "SELECT * FROM users WHERE id = ?", [user_id]
        )
        if not results:
            return None
        return results[0]

    def get_active_users(self):
        results = self.db.query(
            "SELECT * FROM users WHERE active = ?", [True]
        )
        return results

    def count_users(self):
        results = self.db.query("SELECT COUNT(*) FROM users")
        return results[0]["count"]
```

Write tests that mock the `Database` to test all three methods of `UserService`. Verify both the return values AND that the correct SQL was passed.

<details>
<summary><strong>Solution 2.1</strong></summary>

```python
# test_user_service.py
from unittest.mock import MagicMock
from user_service import UserService

def test_get_user_found():
    mock_db = MagicMock()
    mock_db.query.return_value = [{"id": 1, "name": "Alice", "active": True}]

    service = UserService(mock_db)
    user = service.get_user(1)

    assert user == {"id": 1, "name": "Alice", "active": True}
    mock_db.query.assert_called_with(
        "SELECT * FROM users WHERE id = ?", [1]
    )

def test_get_user_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value = []

    service = UserService(mock_db)
    assert service.get_user(999) is None

def test_get_active_users():
    mock_db = MagicMock()
    mock_db.query.return_value = [
        {"id": 1, "name": "Alice", "active": True},
        {"id": 3, "name": "Charlie", "active": True},
    ]

    service = UserService(mock_db)
    users = service.get_active_users()

    assert len(users) == 2
    mock_db.query.assert_called_with(
        "SELECT * FROM users WHERE active = ?", [True]
    )

def test_count_users():
    mock_db = MagicMock()
    mock_db.query.return_value = [{"count": 42}]

    service = UserService(mock_db)
    assert service.count_users() == 42
    mock_db.query.assert_called_with("SELECT COUNT(*) FROM users")
```
</details>

---

# WEEKS 3–4: Design Patterns & OOP

## Week 3: Refactoring & OOP Fundamentals

### Concepts
- **Code smells**: magic numbers, global variables, long functions, repeated code
- **Refactoring**: small, behaviour-preserving transformations
- **Extract variable/function**: replace inline expressions with named ones
- **Replace magic number with config**: use YAML/JSON config files
- **Encapsulate in class**: group related data and functions
- **Inheritance vs composition**: prefer composition ("has-a" over "is-a")

### Tutorial 3.1: Refactoring Procedural Code

Before — messy procedural code:

```python
# boids_bad.py
import random

# Bad: magic numbers, global state, one giant function
positions = [[random.uniform(-450, 50) for _ in range(2)] for _ in range(50)]
velocities = [[random.uniform(0, 10) for _ in range(2)] for _ in range(50)]

def update():
    for i in range(50):
        # Fly towards middle
        for j in range(2):
            velocities[i][j] += (sum(p[j] for p in positions) / 50 - positions[i][j]) * 0.01

        # Fly away from nearby boids
        for other in range(50):
            if other == i:
                continue
            dist = sum((positions[i][j] - positions[other][j])**2 for j in range(2))
            if dist < 100:  # magic number!
                for j in range(2):
                    velocities[i][j] += positions[i][j] - positions[other][j]

        # Match velocity with nearby boids
        for other in range(50):
            if other == i:
                continue
            dist = sum((positions[i][j] - positions[other][j])**2 for j in range(2))
            if dist < 10000:  # another magic number
                for j in range(2):
                    velocities[i][j] += (velocities[other][j] - velocities[i][j]) * 0.125

        # Update position
        for j in range(2):
            positions[i][j] += velocities[i][j]
```

After — refactored step by step:

```python
# boids_refactored.py
import yaml
import numpy as np

class BoidConfig:
    """Configuration extracted from magic numbers."""
    def __init__(self, config_path="boids_config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.count = config["count"]
        self.move_to_middle_strength = config["move_to_middle_strength"]
        self.alert_distance = config["alert_distance"]
        self.formation_flying_distance = config["formation_flying_distance"]
        self.formation_flying_strength = config["formation_flying_strength"]

class Flock:
    """Encapsulates boid simulation state and behaviour."""

    def __init__(self, config):
        self.config = config
        self.positions = np.random.uniform(-450, 50, (config.count, 2))
        self.velocities = np.random.uniform(0, 10, (config.count, 2))

    def _fly_towards_middle(self):
        """Each boid steers towards the flock's centre of mass."""
        centre = self.positions.mean(axis=0)
        direction_to_centre = centre - self.positions
        self.velocities += direction_to_centre * self.config.move_to_middle_strength

    def _fly_away_from_nearby(self):
        """Boids avoid getting too close to each other."""
        for i in range(self.config.count):
            separations = self.positions[i] - self.positions
            squared_distances = (separations ** 2).sum(axis=1)
            too_close = squared_distances < self.config.alert_distance
            too_close[i] = False  # don't avoid yourself
            self.velocities[i] += separations[too_close].sum(axis=0)

    def _match_velocity(self):
        """Boids align velocity with nearby neighbours."""
        for i in range(self.config.count):
            separations = self.positions[i] - self.positions
            squared_distances = (separations ** 2).sum(axis=1)
            nearby = squared_distances < self.config.formation_flying_distance
            nearby[i] = False
            if nearby.any():
                mean_velocity = self.velocities[nearby].mean(axis=0)
                self.velocities[i] += (
                    (mean_velocity - self.velocities[i])
                    * self.config.formation_flying_strength
                )

    def update(self):
        """Single simulation step: apply all rules then move."""
        self._fly_towards_middle()
        self._fly_away_from_nearby()
        self._match_velocity()
        self.positions += self.velocities
```

And the config file:

```yaml
# boids_config.yaml
count: 50
move_to_middle_strength: 0.01
alert_distance: 100
formation_flying_distance: 10000
formation_flying_strength: 0.125
```

**Key refactorings applied:**
1. Magic numbers → config file
2. Global variables → instance attributes
3. One big function → small methods with clear names
4. Parallel lists → NumPy arrays
5. Procedural → OOP with `Flock` class

### Exercise 3.1: Refactor This Code

Refactor the following into a well-structured class with config:

```python
# particle_sim_bad.py
import random
import math

particles_x = [random.uniform(0, 100) for _ in range(30)]
particles_y = [random.uniform(0, 100) for _ in range(30)]
particles_mass = [random.uniform(1, 10) for _ in range(30)]

def step():
    for i in range(30):
        fx, fy = 0, 0
        for j in range(30):
            if i == j:
                continue
            dx = particles_x[j] - particles_x[i]
            dy = particles_y[j] - particles_y[i]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 0.1:
                dist = 0.1
            force = 6.674e-11 * particles_mass[i] * particles_mass[j] / (dist * dist)
            fx += force * dx / dist
            fy += force * dy / dist
        particles_x[i] += fx / particles_mass[i] * 0.01
        particles_y[i] += fy / particles_mass[i] * 0.01
```

Your refactored version should have: a `Particle` dataclass or class, a `Simulation` class, config for magic numbers (G, dt, min_distance, count, bounds), and small named methods.

<details>
<summary><strong>Solution 3.1</strong></summary>

```python
# particle_sim.py
from dataclasses import dataclass, field
import numpy as np
import yaml

@dataclass
class SimConfig:
    count: int = 30
    bounds: float = 100.0
    mass_range: tuple = (1.0, 10.0)
    gravitational_constant: float = 6.674e-11
    time_step: float = 0.01
    min_distance: float = 0.1

class ParticleSimulation:
    def __init__(self, config: SimConfig = None):
        self.config = config or SimConfig()
        c = self.config
        self.positions = np.random.uniform(0, c.bounds, (c.count, 2))
        self.masses = np.random.uniform(*c.mass_range, c.count)

    def _compute_forces(self):
        """Compute gravitational force on each particle from all others."""
        forces = np.zeros_like(self.positions)
        for i in range(self.config.count):
            displacements = self.positions - self.positions[i]  # (N, 2)
            distances = np.linalg.norm(displacements, axis=1)   # (N,)
            distances[i] = np.inf  # ignore self
            distances = np.maximum(distances, self.config.min_distance)

            force_magnitudes = (
                self.config.gravitational_constant
                * self.masses[i] * self.masses
                / distances ** 2
            )
            # Unit direction vectors
            directions = displacements / distances[:, np.newaxis]
            directions[i] = 0

            forces[i] = (force_magnitudes[:, np.newaxis] * directions).sum(axis=0)
        return forces

    def step(self):
        """Advance simulation by one time step."""
        forces = self._compute_forces()
        accelerations = forces / self.masses[:, np.newaxis]
        self.positions += accelerations * self.config.time_step
```
</details>

---

## Week 4: Design Patterns

### Concepts
- **Factory**: defer object creation to subclasses or factory functions
- **Builder**: construct complex objects step by step
- **Strategy**: swap algorithms at runtime via interchangeable objects
- **Observer**: notify dependents automatically when state changes
- **Composition over inheritance**: prefer "has-a" relationships

### Tutorial 4.1: Factory Pattern

From the course — handling different image instrument formats uniformly:

```python
# factory_example.py
from abc import ABC, abstractmethod

class ImageLoader(ABC):
    """Abstract interface for loading images from different instruments."""

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def metadata(self):
        pass

class MicroscopeImage(ImageLoader):
    def load(self, path):
        print(f"Loading microscope TIFF from {path}")
        self.data = f"microscope_data_{path}"

    def metadata(self):
        return {"instrument": "microscope", "format": "tiff"}

class TelescopeImage(ImageLoader):
    def load(self, path):
        print(f"Loading telescope FITS from {path}")
        self.data = f"telescope_data_{path}"

    def metadata(self):
        return {"instrument": "telescope", "format": "fits"}

class SatelliteImage(ImageLoader):
    def load(self, path):
        print(f"Loading satellite HDF5 from {path}")
        self.data = f"satellite_data_{path}"

    def metadata(self):
        return {"instrument": "satellite", "format": "hdf5"}

# The factory function
def create_image_loader(instrument_type):
    """Factory: create the right loader based on instrument type."""
    loaders = {
        "microscope": MicroscopeImage,
        "telescope": TelescopeImage,
        "satellite": SatelliteImage,
    }
    if instrument_type not in loaders:
        raise ValueError(f"Unknown instrument: {instrument_type}")
    return loaders[instrument_type]()

# Usage — client code doesn't need to know which class
loader = create_image_loader("telescope")
loader.load("galaxy_m31.fits")
print(loader.metadata())
```

### Tutorial 4.2: Builder Pattern

```python
# builder_example.py

class SimulationModel:
    """A complex simulation that requires multiple configuration steps."""

    def __init__(self):
        self.grid = None
        self.boundary_conditions = None
        self.initial_conditions = None
        self.solver = None

    def run(self):
        # Validate everything is set
        for attr in ["grid", "boundary_conditions", "initial_conditions", "solver"]:
            if getattr(self, attr) is None:
                raise RuntimeError(f"{attr} not set — use the builder")
        print(f"Running with {self.solver} on {self.grid} grid")

class SimulationBuilder:
    """Builder: each configuration step is a method, finish() validates."""

    def __init__(self):
        self._model = SimulationModel()

    def set_grid(self, resolution, dimensions):
        self._model.grid = {"resolution": resolution, "dimensions": dimensions}
        return self  # allow chaining

    def set_boundary(self, bc_type):
        self._model.boundary_conditions = bc_type
        return self

    def set_initial_conditions(self, ic_func):
        self._model.initial_conditions = ic_func
        return self

    def set_solver(self, solver_name):
        self._model.solver = solver_name
        return self

    def finish(self):
        """Validate and return the completed model."""
        model = self._model
        for attr in ["grid", "boundary_conditions", "initial_conditions", "solver"]:
            if getattr(model, attr) is None:
                raise RuntimeError(f"Cannot build: {attr} not configured")
        return model

# Usage with chaining
model = (
    SimulationBuilder()
    .set_grid(100, 2)
    .set_boundary("periodic")
    .set_initial_conditions(lambda x: x**2)
    .set_solver("runge-kutta-4")
    .finish()
)
model.run()
```

### Tutorial 4.3: Strategy Pattern

```python
# strategy_example.py
from abc import ABC, abstractmethod

class IntegrationStrategy(ABC):
    @abstractmethod
    def integrate(self, f, a, b, n):
        pass

class TrapezoidRule(IntegrationStrategy):
    def integrate(self, f, a, b, n):
        h = (b - a) / n
        result = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            result += f(a + i * h)
        return result * h

class SimpsonRule(IntegrationStrategy):
    def integrate(self, f, a, b, n):
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        result = f(a) + f(b)
        for i in range(1, n, 2):
            result += 4 * f(a + i * h)
        for i in range(2, n, 2):
            result += 2 * f(a + i * h)
        return result * h / 3

class NumericalIntegrator:
    """Context: uses a strategy to perform integration."""

    def __init__(self, strategy: IntegrationStrategy):
        self.strategy = strategy

    def compute(self, f, a, b, n=1000):
        return self.strategy.integrate(f, a, b, n)

# Usage — swap algorithms without changing client code
import math

integrator = NumericalIntegrator(TrapezoidRule())
result = integrator.compute(math.sin, 0, math.pi)
print(f"Trapezoid: {result:.6f}")

integrator.strategy = SimpsonRule()
result = integrator.compute(math.sin, 0, math.pi)
print(f"Simpson:   {result:.6f}")
# Both should be close to 2.0
```

### Exercise 4.1: Implement an Observer Pattern

Build a data pipeline where a `DataSource` notifies multiple `Observer` objects whenever new data arrives. Requirements:

1. `DataSource` has `add_observer()`, `remove_observer()`, and `notify()` methods
2. Create `Logger` (prints to console), `StatisticsTracker` (maintains running mean), and `Threshold` (raises alert if value exceeds limit) observers
3. All observers implement an `update(value)` method
4. Write tests for the whole thing

<details>
<summary><strong>Solution 4.1</strong></summary>

```python
# observer.py
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, value):
        pass

class DataSource:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify(self, value):
        for observer in self._observers:
            observer.update(value)

    def new_data(self, value):
        """Simulate receiving new data."""
        self.notify(value)

class Logger(Observer):
    def __init__(self):
        self.log = []

    def update(self, value):
        self.log.append(value)

class StatisticsTracker(Observer):
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    @property
    def mean(self):
        return sum(self.values) / len(self.values) if self.values else 0

    @property
    def count(self):
        return len(self.values)

class ThresholdAlert(Observer):
    def __init__(self, limit):
        self.limit = limit
        self.alerts = []

    def update(self, value):
        if value > self.limit:
            self.alerts.append(value)

# test_observer.py
import pytest
from observer import DataSource, Logger, StatisticsTracker, ThresholdAlert

def test_logger_receives_all_values():
    source = DataSource()
    logger = Logger()
    source.add_observer(logger)

    source.new_data(10)
    source.new_data(20)
    source.new_data(30)

    assert logger.log == [10, 20, 30]

def test_statistics_tracker():
    source = DataSource()
    tracker = StatisticsTracker()
    source.add_observer(tracker)

    for v in [10, 20, 30]:
        source.new_data(v)

    assert tracker.count == 3
    assert tracker.mean == pytest.approx(20.0)

def test_threshold_alert():
    source = DataSource()
    alert = ThresholdAlert(limit=25)
    source.add_observer(alert)

    for v in [10, 20, 30, 5, 50]:
        source.new_data(v)

    assert alert.alerts == [30, 50]

def test_remove_observer():
    source = DataSource()
    logger = Logger()
    source.add_observer(logger)
    source.new_data(1)
    source.remove_observer(logger)
    source.new_data(2)

    assert logger.log == [1]

def test_multiple_observers():
    source = DataSource()
    logger = Logger()
    tracker = StatisticsTracker()
    alert = ThresholdAlert(100)

    source.add_observer(logger)
    source.add_observer(tracker)
    source.add_observer(alert)

    source.new_data(50)
    source.new_data(150)

    assert logger.log == [50, 150]
    assert tracker.mean == pytest.approx(100.0)
    assert alert.alerts == [150]
```
</details>

---

# WEEKS 5–6: Advanced Programming Techniques

## Week 5: Iterators, Generators & Decorators

### Concepts
- **Iterable**: any object with `__iter__()` returning an iterator
- **Iterator**: object with `__next__()` (raises `StopIteration` when done)
- **Generator**: function using `yield` — creates iterators concisely
- **Decorator**: function that wraps another function to add behaviour
- **Context manager**: `__enter__`/`__exit__` protocol (or `@contextmanager`)

### Tutorial 5.1: Building Iterators and Generators

```python
# iterators_demo.py

# --- Iterator class ---
class Countdown:
    """Iterator that counts down from n to 1."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        value = self.n
        self.n -= 1
        return value

# Usage:
for x in Countdown(5):
    print(x)  # 5, 4, 3, 2, 1

# --- Generator function (much simpler!) ---
def countdown(n):
    """Generator that counts down from n to 1."""
    while n > 0:
        yield n
        n -= 1

for x in countdown(5):
    print(x)  # same output

# --- Generator for infinite sequences ---
def fibonacci():
    """Infinite Fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Take first 10
from itertools import islice
first_10 = list(islice(fibonacci(), 10))
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# --- Generator for file processing (memory efficient) ---
def read_large_file(filepath):
    """Yield lines one at a time without loading entire file."""
    with open(filepath) as f:
        for line in f:
            yield line.strip()

def filter_lines(lines, keyword):
    """Generator that filters lines containing keyword."""
    for line in lines:
        if keyword in line:
            yield line

# Chaining generators — processes one line at a time
# matching = filter_lines(read_large_file("huge.log"), "ERROR")
```

### Tutorial 5.2: Writing Decorators

```python
# decorators_demo.py
import time
import functools

# --- Basic decorator ---
def timer(func):
    """Measure execution time of a function."""
    @functools.wraps(func)  # preserves original function's name/docstring
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.1)
    return 42

result = slow_function()  # prints: slow_function took 0.100Xs

# --- Decorator with arguments ---
def retry(max_attempts=3):
    """Retry a function up to max_attempts times on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying...")
        return wrapper
    return decorator

@retry(max_attempts=3)
def flaky_operation():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Network timeout")
    return "Success!"

# --- Decorator for input validation ---
def validate_positive(func):
    """Ensure all numeric arguments are positive."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"Expected positive value, got {arg}")
        return func(*args, **kwargs)
    return wrapper

@validate_positive
def compute_area(width, height):
    return width * height
```

### Tutorial 5.3: Context Managers

```python
# context_managers_demo.py
from contextlib import contextmanager
import time

# --- Class-based context manager ---
class Timer:
    """Context manager to time a block of code."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        print(f"Block took {self.elapsed:.4f}s")
        return False  # don't suppress exceptions

with Timer() as t:
    time.sleep(0.1)
# prints: Block took 0.100Xs

# --- Generator-based context manager (simpler) ---
@contextmanager
def timer_context():
    start = time.time()
    yield  # code in the 'with' block runs here
    elapsed = time.time() - start
    print(f"Block took {elapsed:.4f}s")

with timer_context():
    time.sleep(0.1)

# --- Practical: temporary directory context ---
@contextmanager
def working_directory(path):
    """Temporarily change working directory, restore on exit."""
    import os
    original = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(original)  # always restore, even on exception
```

### Exercise 5.1: Build a Mini Test Framework

The course shows how generators and context managers combine to build test frameworks. Build one:

1. Write a `@test_case` decorator that registers functions in a test registry
2. Write a `run_tests()` function that runs all registered tests and reports pass/fail
3. Write a context manager `expect_raises(exception_type)` that passes if the block raises the expected exception
4. Add a `@timed(max_seconds)` decorator that fails the test if it takes too long

<details>
<summary><strong>Solution 5.1</strong></summary>

```python
# mini_test_framework.py
import functools
import time
from contextlib import contextmanager

_test_registry = []

def test_case(func):
    """Decorator: register a function as a test case."""
    _test_registry.append(func)
    return func

def timed(max_seconds):
    """Decorator: fail if function takes longer than max_seconds."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if elapsed > max_seconds:
                raise TimeoutError(
                    f"{func.__name__} took {elapsed:.3f}s "
                    f"(limit: {max_seconds}s)"
                )
            return result
        return wrapper
    return decorator

@contextmanager
def expect_raises(exception_type):
    """Context manager: pass if block raises expected exception."""
    try:
        yield
    except exception_type:
        return  # expected exception occurred — pass
    except Exception as e:
        raise AssertionError(
            f"Expected {exception_type.__name__}, "
            f"got {type(e).__name__}: {e}"
        )
    else:
        raise AssertionError(
            f"Expected {exception_type.__name__} but no exception was raised"
        )

def run_tests():
    """Run all registered test cases and report results."""
    passed = 0
    failed = 0
    for test_func in _test_registry:
        try:
            test_func()
            print(f"  PASS: {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_func.__name__} — {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")

# --- Example usage ---

@test_case
def test_addition():
    assert 2 + 2 == 4

@test_case
def test_negative_testing():
    with expect_raises(ZeroDivisionError):
        1 / 0

@test_case
@timed(max_seconds=0.5)
def test_fast_enough():
    total = sum(range(100000))
    assert total > 0

@test_case
def test_expect_raises_fails_correctly():
    """This test SHOULD fail to demonstrate the framework."""
    with expect_raises(ValueError):
        pass  # no exception raised — should fail

if __name__ == "__main__":
    run_tests()
```
</details>

---

## Week 6: Operator Overloading & Metaprogramming

### Concepts
- **Operator overloading**: `__add__`, `__mul__`, `__eq__`, `__repr__`, `__str__`, etc.
- **Metaprogramming**: modifying classes/modules programmatically at runtime
- `globals()` and `__dict__` for accessing/modifying namespace
- `setattr()` and `getattr()` for dynamic attribute access
- **Duck typing**: "if it quacks like a duck" — rely on interfaces, not types
- **KISS principle**: don't overuse metaprogramming when simpler solutions exist

### Tutorial 6.1: Operator Overloading — Symbolic Algebra

From the course's operator overloading example:

```python
# algebra.py

class Term:
    """Represents a term like 3x^2*y^3."""

    def __init__(self, coefficient=1, **variables):
        self.coefficient = coefficient
        self.variables = {k: v for k, v in variables.items() if v != 0}

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Term(
                coefficient=self.coefficient * other,
                **self.variables
            )
        if isinstance(other, Term):
            new_vars = dict(self.variables)
            for var, power in other.variables.items():
                new_vars[var] = new_vars.get(var, 0) + power
            return Term(
                coefficient=self.coefficient * other.coefficient,
                **new_vars
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        parts = [str(self.coefficient)] if self.coefficient != 1 else []
        for var, power in sorted(self.variables.items()):
            if power == 1:
                parts.append(var)
            else:
                parts.append(f"{var}^{power}")
        return " * ".join(parts) if parts else "1"

    def __eq__(self, other):
        if not isinstance(other, Term):
            return NotImplemented
        return (
            self.coefficient == other.coefficient
            and self.variables == other.variables
        )

class Expression:
    """Sum of terms, e.g. 3x^2 + 2xy - 1."""

    def __init__(self, terms=None):
        self.terms = terms or []

    def __add__(self, other):
        if isinstance(other, Term):
            return Expression(self.terms + [other])
        if isinstance(other, Expression):
            return Expression(self.terms + other.terms)
        return NotImplemented

    def __repr__(self):
        return " + ".join(repr(t) for t in self.terms)

# Usage
x = Term(x=1)       # x
y = Term(y=1)       # y
three_x_sq = 3 * x * x   # 3x^2
two_xy = 2 * x * y       # 2xy

expr = Expression([three_x_sq]) + two_xy
print(expr)  # 3 * x^2 + 2 * x * y
```

### Tutorial 6.2: Metaprogramming

```python
# metaprogramming_demo.py

# --- Dynamic attribute creation with setattr ---
class DataRecord:
    pass

# Programmatically add attributes from a dictionary
fields = {"name": "Alice", "age": 30, "role": "engineer"}
record = DataRecord()
for key, value in fields.items():
    setattr(record, key, value)

print(record.name)  # Alice
print(record.age)   # 30

# --- Programmatically adding methods to a class ---
def make_getter(field_name):
    def getter(self):
        return getattr(self, f"_{field_name}")
    getter.__name__ = f"get_{field_name}"
    return getter

def make_setter(field_name):
    def setter(self, value):
        setattr(self, f"_{field_name}", value)
    setter.__name__ = f"set_{field_name}"
    return setter

class AutoProperties:
    _fields = ["x", "y", "z"]

for field in AutoProperties._fields:
    setattr(AutoProperties, f"get_{field}", make_getter(field))
    setattr(AutoProperties, f"set_{field}", make_setter(field))

obj = AutoProperties()
obj.set_x(10)
obj.set_y(20)
print(obj.get_x())  # 10

# --- The KISS principle: often a dict is better ---
# Instead of metaprogramming:
baskets = {}
for fruit in ["apples", "bananas", "oranges"]:
    baskets[fruit] = 0

# This is cleaner than:
# for fruit in ["apples", "bananas", "oranges"]:
#     globals()[fruit] = 0  # <-- works but don't do this
```

### Exercise 6.1: Build a Vector Class

Create a `Vector` class that supports:
- `+` and `-` (element-wise)
- `*` (scalar multiplication and dot product)
- `==` for comparison
- `len()` for dimension
- `abs()` for magnitude
- `repr` showing e.g. `Vector(1, 2, 3)`
- Iteration (can use in `for` loops)
- Indexing with `[]`

Write tests for every operation.

<details>
<summary><strong>Solution 6.1</strong></summary>

```python
# vector.py
import math

class Vector:
    def __init__(self, *components):
        self._data = list(components)

    def __repr__(self):
        return f"Vector({', '.join(str(c) for c in self._data)})"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __iter__(self):
        return iter(self._data)

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self._data == other._data

    def __add__(self, other):
        if not isinstance(other, Vector) or len(self) != len(other):
            return NotImplemented
        return Vector(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other):
        if not isinstance(other, Vector) or len(self) != len(other):
            return NotImplemented
        return Vector(*(a - b for a, b in zip(self, other)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(*(c * other for c in self))
        if isinstance(other, Vector):
            if len(self) != len(other):
                return NotImplemented
            return sum(a * b for a, b in zip(self, other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __abs__(self):
        return math.sqrt(sum(c ** 2 for c in self))

# test_vector.py
import pytest
from pytest import approx
from vector import Vector

def test_repr():
    assert repr(Vector(1, 2, 3)) == "Vector(1, 2, 3)"

def test_len():
    assert len(Vector(1, 2, 3)) == 3

def test_getitem():
    v = Vector(10, 20, 30)
    assert v[0] == 10
    assert v[2] == 30

def test_iter():
    assert list(Vector(1, 2, 3)) == [1, 2, 3]

def test_eq():
    assert Vector(1, 2) == Vector(1, 2)
    assert Vector(1, 2) != Vector(1, 3)

def test_add():
    assert Vector(1, 2) + Vector(3, 4) == Vector(4, 6)

def test_sub():
    assert Vector(5, 3) - Vector(1, 1) == Vector(4, 2)

def test_scalar_mul():
    assert Vector(1, 2, 3) * 2 == Vector(2, 4, 6)
    assert 3 * Vector(1, 2) == Vector(3, 6)

def test_dot_product():
    assert Vector(1, 2, 3) * Vector(4, 5, 6) == 32

def test_abs():
    assert abs(Vector(3, 4)) == approx(5.0)

def test_zero_vector():
    assert abs(Vector(0, 0, 0)) == 0.0
```
</details>

---

# WEEKS 7–8: Performance & Packaging

## Week 7: Performance Optimisation

### Concepts
- **Profiling before optimising**: never guess where the bottleneck is
- `%timeit` for quick benchmarks, `%prun` for function-level profiling
- `line_profiler` for line-by-line profiling
- **NumPy vectorisation**: replace Python loops with array operations
- **Scaling laws**: O(1), O(N), O(N²), O(log N) — and how data structures affect this
- **Cython**: add type annotations to Python for C-level speed
- List append is O(1), list insert is O(N); NumPy append is O(N), lookup is O(1)
- Dict lookup is O(1) via hash tables

### Tutorial 7.1: Profiling and Vectorising

```python
# mandelbrot_slow.py
"""Naive Mandelbrot — pure Python loops."""

def mandel_slow(position, limit=50):
    value = position
    while abs(value) < 2:
        limit -= 1
        value = value ** 2 + position
        if limit < 0:
            return 0
    return limit

def compute_mandelbrot_slow(xmin, xmax, ymin, ymax, resolution=300, limit=50):
    """Pure Python: nested loops over every pixel."""
    import numpy as np
    xstep = (xmax - xmin) / resolution
    ystep = (ymax - ymin) / resolution
    result = []
    for y_idx in range(resolution):
        row = []
        for x_idx in range(resolution):
            x = xmin + x_idx * xstep
            y = ymin + y_idx * ystep
            row.append(mandel_slow(complex(x, y), limit))
        result.append(row)
    return result
```

Now the NumPy-vectorised version:

```python
# mandelbrot_numpy.py
"""Vectorised Mandelbrot using NumPy."""
import numpy as np

def compute_mandelbrot_numpy(xmin, xmax, ymin, ymax, resolution=300, limit=50):
    xstep = (xmax - xmin) / resolution
    ystep = (ymax - ymin) / resolution

    # Create grid of complex numbers all at once
    y, x = np.mgrid[ymin:ymax:ystep, xmin:xmax:xstep]
    c = x + 1j * y

    # Iterate ALL points simultaneously
    z = np.zeros_like(c)
    diverged_at = np.zeros(c.shape, dtype=int)

    for i in range(limit):
        mask = np.abs(z) <= 2  # which points haven't escaped
        z[mask] = z[mask] ** 2 + c[mask]
        diverged_at[mask] = i

    return diverged_at
```

Benchmark them:

```python
# benchmark.py
import time

# Slow version
start = time.time()
result_slow = compute_mandelbrot_slow(-2, 1, -1.5, 1.5, resolution=100)
print(f"Pure Python:  {time.time() - start:.3f}s")

# Fast version
start = time.time()
result_fast = compute_mandelbrot_numpy(-2, 1, -1.5, 1.5, resolution=100)
print(f"NumPy:        {time.time() - start:.3f}s")

# You should see 10-50x speedup
```

### Tutorial 7.2: Scaling Laws

```python
# scaling_demo.py
"""Demonstrate O(1) vs O(N) behaviour."""
import time
import numpy as np

def time_operation(operation, sizes, repeats=100):
    """Measure execution time across different input sizes."""
    times = []
    for n in sizes:
        total = 0
        for _ in range(repeats):
            start = time.perf_counter()
            operation(n)
            total += time.perf_counter() - start
        times.append(total / repeats)
    return times

# O(1): list append
def list_append(n):
    data = [0] * n
    data.append(1)

# O(N): list insert at beginning
def list_insert(n):
    data = [0] * n
    data.insert(0, 1)

# O(1): dict lookup
def dict_lookup(n):
    data = {i: i for i in range(n)}
    _ = data.get(n // 2)

# O(N): search in list
def list_search(n):
    data = list(range(n))
    _ = (n // 2) in data

sizes = [1000, 5000, 10000, 50000, 100000]

print("List append (O(1)):", time_operation(list_append, sizes))
print("List insert (O(N)):", time_operation(list_insert, sizes))
print("Dict lookup (O(1)):", time_operation(dict_lookup, sizes))
print("List search (O(N)):", time_operation(list_search, sizes))
```

### Tutorial 7.3: Cython Basics

In a Jupyter notebook:

```python
%load_ext Cython
```

```cython
%%cython
# Pure Python in Cython (no type annotations) — already slightly faster
def mandel_cython_untyped(position, limit=50):
    value = position
    while abs(value) < 2:
        limit -= 1
        value = value**2 + position
        if limit < 0:
            return 0
    return limit
```

```cython
%%cython
# Typed Cython — much faster
cpdef int mandel_cython_typed(double complex position, int limit=50):
    cdef double complex value
    value = position
    while abs(value) < 2:
        limit -= 1
        value = value**2 + position
        if limit < 0:
            return 0
    return limit
```

```python
# Compare
%timeit mandel_slow(complex(-0.5, 0.5))
%timeit mandel_cython_untyped(complex(-0.5, 0.5))
%timeit mandel_cython_typed(complex(-0.5, 0.5))
# Expect: untyped ~2x faster, typed ~10-50x faster than pure Python
```

### Exercise 7.1: Optimise a Pairwise Distance Calculator

This function computes all pairwise distances between N points. Optimise it.

```python
# distances_slow.py
import math

def pairwise_distances_slow(points):
    """
    Compute NxN matrix of Euclidean distances.
    points: list of (x, y) tuples
    """
    n = len(points)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            result[i][j] = math.sqrt(dx*dx + dy*dy)
    return result
```

Write three versions:
1. NumPy vectorised (no explicit loops)
2. NumPy with broadcasting
3. (Optional) Cython with type annotations

Benchmark all versions with 500 points.

<details>
<summary><strong>Solution 7.1</strong></summary>

```python
# distances_fast.py
import numpy as np

def pairwise_distances_numpy(points):
    """Vectorised with broadcasting."""
    pts = np.array(points)  # (N, 2)
    # diff[i, j, :] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=2))  # (N, N)

def pairwise_distances_scipy(points):
    """Using scipy — the practical choice."""
    from scipy.spatial.distance import cdist
    pts = np.array(points)
    return cdist(pts, pts)

# Benchmark
import time

N = 500
points = [(np.random.rand(), np.random.rand()) for _ in range(N)]
pts_array = np.random.rand(N, 2)

start = time.time()
r1 = pairwise_distances_slow(points)
print(f"Pure Python: {time.time() - start:.3f}s")

start = time.time()
r2 = pairwise_distances_numpy(pts_array.tolist())
print(f"NumPy broadcast: {time.time() - start:.3f}s")

start = time.time()
r3 = pairwise_distances_scipy(pts_array.tolist())
print(f"SciPy cdist: {time.time() - start:.3f}s")
```
</details>

---

## Week 8: Packaging & Revision

### Concepts
- Python packages: `__init__.py`, module structure
- `pyproject.toml`: modern Python packaging configuration
- Entry points / console scripts via `[project.scripts]`
- `argparse`: command-line argument parsing
- Installing in development mode: `pip install -e .`
- Choosing a license: MIT, GPL, BSD, Apache

### Tutorial 8.1: Building a Package from Scratch

Directory structure:

```
my_stats/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── my_stats/
│       ├── __init__.py
│       ├── descriptive.py
│       └── cli.py
└── tests/
    ├── __init__.py
    └── test_descriptive.py
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my-stats"
version = "0.1.0"
description = "A simple statistics library"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = []

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
my-stats = "my_stats.cli:main"
```

```python
# src/my_stats/__init__.py
from .descriptive import mean, variance, std_dev, median
```

```python
# src/my_stats/descriptive.py
import math

def mean(data):
    if not data:
        raise ValueError("Cannot compute mean of empty sequence")
    return sum(data) / len(data)

def variance(data):
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)

def std_dev(data):
    return math.sqrt(variance(data))

def median(data):
    if not data:
        raise ValueError("Cannot compute median of empty sequence")
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
```

```python
# src/my_stats/cli.py
import argparse
from . import mean, variance, std_dev, median

def main():
    parser = argparse.ArgumentParser(description="Compute statistics")
    parser.add_argument("numbers", nargs="+", type=float,
                        help="Numbers to analyse")
    parser.add_argument("--stat", choices=["mean", "var", "std", "median", "all"],
                        default="all", help="Which statistic to compute")

    args = parser.parse_args()
    data = args.numbers

    stats = {
        "mean": mean,
        "var": variance,
        "std": std_dev,
        "median": median,
    }

    if args.stat == "all":
        for name, func in stats.items():
            try:
                print(f"{name}: {func(data):.4f}")
            except ValueError as e:
                print(f"{name}: {e}")
    else:
        func = stats[args.stat]
        print(f"{func(data):.4f}")

if __name__ == "__main__":
    main()
```

Install and test:

```bash
pip install -e ".[dev]"
pytest tests/
my-stats 1 2 3 4 5 --stat all
```

### Exercise 8.1: Package the Vector Class

Take your `Vector` class from Exercise 6.1 and turn it into a proper installable package:

1. Create the package structure with `pyproject.toml`
2. Add a CLI that reads vectors from a file (one per line, space-separated components) and computes their sum
3. Include your tests, runnable with `pytest`
4. Add a `README.md`

---

# Quick Reference: What to Revise the Night Before

**Testing**
- `assert`, `pytest.approx(val, rel=1e-6)`, `pytest.raises(Exception)`
- `@pytest.mark.parametrize("args", [(...), (...)])`
- `unittest.mock.patch`, `MagicMock`, `assert_called_with`, `assert_any_call`
- `@pytest.fixture`, `tmp_path`

**OOP & Patterns**
- Factory: function returns different class instances based on input
- Builder: chain `.set_x().set_y().finish()` — validates on finish
- Strategy: interchangeable algorithm objects
- Observer: subject notifies list of observers on change
- Composition > inheritance

**Advanced**
- Iterator: `__iter__` + `__next__` (raise `StopIteration`)
- Generator: `yield` in a function
- Decorator: `def wrapper(func): @functools.wraps(func) def inner(*args, **kwargs)...`
- Context manager: `__enter__` + `__exit__` or `@contextmanager` with `yield`
- Operator overloading: `__add__`, `__mul__`, `__rmul__`, `__eq__`, `__repr__`, `__len__`, `__abs__`, `__getitem__`, `__iter__`
- Metaprogramming: `setattr`, `getattr`, `globals()`, `cls.__dict__`

**Performance**
- `%timeit`, `%prun`, `line_profiler`
- NumPy: vectorise loops, broadcasting, `mgrid`
- Scaling: list append O(1), insert O(N); dict lookup O(1); numpy append O(N)
- Cython: `cdef`, `cpdef`, typed variables → C-speed

**Packaging**
- `pyproject.toml` with `[project]`, `[project.scripts]`, `[build-system]`
- `argparse`: `add_argument`, `nargs`, `choices`, `type`
- `pip install -e .` for development
