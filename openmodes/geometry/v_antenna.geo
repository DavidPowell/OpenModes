// Flat V-shaped antenna, based on design of Capasso, with all ends rounded

// allow the geometric parameters to be specified on the command-line

// the width of each arm
If (!Exists(width))
    width = 5e-3;
EndIf

// the length of each arm
If (!Exists(length))
    length = 20.0e-3;
EndIf

// The angular span between the arm centres in degrees
If (!Exists(span))
    span = 45.0;
EndIf

// The orientation of the centre of the V relative to the x-axis is "orientation"
If (!Exists(orientation))
    orientation = 0.0;
EndIf

// If span is greater than 180 degrees, modify both span and
// orientation so that span is less than 180
If (span > 180.0)
    span = 360.0 - span;
    orientation = orientation + 180.0;
EndIf

// the angle parallel to each arm
first_parallel = (orientation - span*0.5)/180.0*Pi;
second_parallel = (orientation + span*0.5)/180.0*Pi;

// the half width of strips
hw = 0.5*width;

// reduced length of the unrounded parts of strips
rl = length-width;

// the point where the two edges touch inside the V
//touch = 2*hw/Tan(span/180.0*Pi);
touch = hw/Sin(0.5*span/180.0*Pi);

lc = 2e-3;

p = newp-1;
Point(p+1) = {0, 0, 0, lc};
Point(p+2) = {hw*Sin(first_parallel), -hw*Cos(first_parallel), 0.0, lc};
Point(p+3) = {hw*Sin(first_parallel)+rl*Cos(first_parallel), -hw*Cos(first_parallel)+rl*Sin(first_parallel), 0.0, lc};
Point(p+4) = {rl*Cos(first_parallel), rl*Sin(first_parallel), 0.0, lc};
Point(p+5) = {(rl+hw)*Cos(first_parallel), (rl+hw)*Sin(first_parallel), 0.0, lc};
Point(p+6) = {-hw*Sin(first_parallel)+rl*Cos(first_parallel), hw*Cos(first_parallel)+rl*Sin(first_parallel), 0.0, lc};
Point(p+7) = {touch*Cos(orientation/180.0*Pi), touch*Sin(orientation/180.0*Pi), 0, lc};
Point(p+8) = {hw*Sin(second_parallel)+rl*Cos(second_parallel), -hw*Cos(second_parallel)+rl*Sin(second_parallel), 0.0, lc};
Point(p+9) = {rl*Cos(second_parallel), rl*Sin(second_parallel), 0.0, lc};
Point(p+10) = {(rl+hw)*Cos(second_parallel), (rl+hw)*Sin(second_parallel), 0.0, lc};
Point(p+11) = {-hw*Sin(second_parallel)+rl*Cos(second_parallel), hw*Cos(second_parallel)+rl*Sin(second_parallel), 0.0, lc};
Point(p+12) = {-hw*Sin(second_parallel), hw*Cos(second_parallel), 0.0, lc};


l = newl-1;
Line(l+1) = {p+2, p+3};
Circle(l+2) = {p+3, p+4, p+5};
Circle(l+3) = {p+5, p+4, p+6};
Line(l+4) = {p+6, p+7};
Line(l+5) = {p+7, p+8};
Circle(l+6) = {p+8, p+9, p+10};
Circle(l+7) = {p+10, p+9, p+11};
Line(l+8) = {p+11, p+12};
Circle(l+9) = {p+12, p+1, p+2};
Compound Line(l+10) = {l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9};


ll = newll;
//Line Loop(ll) = {l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9};
//Line Loop(ll) = {l+2, l+3, l+4, l+5, l+6, l+7, -(l+10)};
Line Loop(ll) = {l+10};

s = news;
Plane Surface(s) = {ll};
Physical Surface(s+1) = {s};

