
// RUN : ls

int foo(int x) { return 1; }

int main() {
  int y = 2;
  int x = y + 1;
  x = y + 1;
  y = x + y;
  if (x > 1) {
    x = y + 1;
  } else {
    y = 2;
  }
  x = x + x;
  return 1;
}

// CHECK: x