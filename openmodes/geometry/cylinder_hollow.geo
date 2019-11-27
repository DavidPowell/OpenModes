// A cylinder with a hole, which may be off-centre

SetFactory("OpenCASCADE");

Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = mesh_tol;
Mesh.CharacteristicLengthMax = mesh_tol;

// inner radius
If (!Exists(inner_radius))
    inner_radius = 1e-3;
EndIf    
    
// outer radius
If (!Exists(outer_radius))
    outer_radius = 2e-3;
EndIf

// radius of filleting edges
If (!Exists(rounding))
    rounding = 0.2e-3;
EndIf

// height of cylinder
If (!Exists(height))
    height = 4e-3;
EndIf

// offset of inner hole from centre
If (!Exists(offset))
    offset = 0.2e-3;
EndIf

Cylinder(1) = {0,0,-0.5*height, 0,0,height, outer_radius};
Cylinder(2) = {offset,0,-0.5*height, 0,0,height, inner_radius};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
Fillet{3}{1,3,4,5}{rounding}

