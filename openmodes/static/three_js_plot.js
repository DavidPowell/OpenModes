function openmodes_three_plot(three_container, json_geo, width, height, initial_wireframe,
                              format_select, arrow_length, skip_webgl) {
    var renderer;
    if ( !skip_webgl && Detector.webgl )
        renderer = new THREE.WebGLRenderer( {antialias:true} );
    else
        renderer = new THREE.CanvasRenderer(); 

    renderer.setSize(width, height);

    var canvas = renderer.domElement;
    three_container.appendChild(canvas);
    canvas.style.cursor = "move";
    
    var scene = new THREE.Scene(),  
        material = new THREE.MeshLambertMaterial( { color: 0xffffff, side: THREE.DoubleSide,
                                                    wireframe: initial_wireframe, vertexColors: THREE.FaceColors  } ),
        geometry = new THREE.Geometry(),
        i, face;
    
    // add the points
    for (i = 0; i < json_geo.nodes.length; i++) { 
        geometry.vertices.push(new THREE.Vector3(json_geo.nodes[i][0], json_geo.nodes[i][1], json_geo.nodes[i][2]));
    }

    if (typeof json_geo.charge === 'undefined') {
        // charge or current data missing, so only display geometry
        for (i = 0; i < json_geo.triangles.length; i++) {
            geometry.faces.push(new THREE.Face3(json_geo.triangles[i][0], json_geo.triangles[i][1], json_geo.triangles[i][2]));
        }

        // hide controls for format of data
        format_select.parentNode.style.visibility="hidden";
        arrow_length.parentNode.style.visibility="hidden";

    } else {
        lut = {};

        // lookup table for magnitude
        lut.abs = new THREE.Lut('cooltowarm', 100);
        var max_abs = Math.max.apply(Math, json_geo.charge.abs);
        lut.abs.setMax(max_abs);
        lut.abs.setMin(-max_abs);

        // lookup table for real and imaginary parts
        lut.real = new THREE.Lut('cooltowarm', 100);
        lut.real.setMax(max_abs);
        lut.real.setMin(-max_abs);
        lut.imag = lut.real;

        // lookup table for phase
        lut.phase = new THREE.Lut('rainbow', 100);
        lut.phase.setMax(180);
        lut.phase.setMin(-180);

        var faces = {}, prop;
        // loop through all the defined data
        for (prop in json_geo.charge) {
            // ignore superfluous metadata
            if (!lut.hasOwnProperty(prop)) continue;

            faces[prop] = [];

            // add the faces, working out the colour of each from the lookup table
            for (i = 0; i < json_geo.triangles.length; i++) {
                face = new THREE.Face3(json_geo.triangles[i][0], json_geo.triangles[i][1], json_geo.triangles[i][2]);
                face.color = lut[prop].getColor(json_geo.charge[prop][i]);
                faces[prop].push(face);
            }
        }

        
            
        
        // the current information is present
        if (typeof json_geo.current !== 'undefined') {
            // current data should be added

            // these variables should only be created if current exists - particularly meshes
            var arrowGeometry, //arrowMesh,
                unitY = new THREE.Vector3(0, 1, 0),
                arrowMeshes = new Array(json_geo.current.real.length), arrowDir,
                currentMax = 0;
            var arrowMaterial = new THREE.MeshLambertMaterial( { color: 0xffffff } );

            for (i = 0; i < json_geo.current.real.length; i++) {
                arrowGeometry = new THREE.CylinderGeometry(0, 1, 1);
                arrowMeshes[i] = new THREE.Mesh( arrowGeometry, arrowMaterial );
                arrowMeshes[i].position.set(json_geo.centres[i][0], json_geo.centres[i][1], json_geo.centres[i][2]);
                arrowDir = new THREE.Vector3(json_geo.current.real[i][1], json_geo.current.real[i][2], json_geo.current.real[i][3]);
                arrowMeshes[i].quaternion.setFromUnitVectors(unitY, arrowDir);
                scene.add( arrowMeshes[i] );
                
                // find the maximum current value
                currentMax = Math.max(currentMax, Math.sqrt(json_geo.current.real[i][0]*json_geo.current.real[i][0] + json_geo.current.real[i][1]*json_geo.current.real[i][1]));
            }
        }
        
        // the function to set the format of the charge and current display
        var updateFormat = function() {
            format = format_select.value;
            geometry.faces = faces[format];
            geometry.colorsNeedUpdate = true;
            var length_i, arrowLengthScale = arrow_length.value/currentMax, arrowWidthScale = arrowLengthScale*0.25;
            
            var current_function = json_geo.current[format];
            if (typeof current_function === 'undefined') {
                for (i = 0; i < json_geo.current.real.length; i++)
                    arrowMeshes[i].visible = false;
            } else {
                for (i = 0; i < json_geo.current.real.length; i++) {
                    length_i = current_function[i][0];
                    arrowDir = new THREE.Vector3(current_function[i][1], current_function[i][2], current_function[i][3]);
                    arrowMeshes[i].quaternion.setFromUnitVectors(unitY, arrowDir);
                    arrowMeshes[i].useQuaternion = true;
                    arrowMeshes[i].scale.set(length_i*arrowWidthScale, length_i*arrowLengthScale, length_i*arrowWidthScale);
                    arrowMeshes[i].visible = true;
                 }
            }
        };
        updateFormat();
        format_select.addEventListener("change", updateFormat);
        arrow_length.addEventListener("change", updateFormat);
    }
    geometry.computeFaceNormals();

    // find centre of geometry, point camera and controls at this
    geometry.computeBoundingSphere()
    var center = geometry.boundingSphere.center;

    var mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    var camera_fov = 20; // camera field of view, in degrees
    var camera = new THREE.PerspectiveCamera(camera_fov, canvas.width / canvas.height, 0.1, 10000);
    camera.position.copy(center)
    camera.position.z += 400;
    camera.lookAt(center);   

    var controls = new THREE.OrbitControls( camera, canvas );
    controls.noKeys = true;
    controls.target0.copy(center);
    controls.reset();
    
    var axisHelper = new THREE.AxisHelper( 20 );
    scene.add( axisHelper );
    
    // create a series of equally spaced points for lighting sources
    tet = new THREE.TetrahedronGeometry(100);
    var mat = new THREE.Matrix4();
    mat.makeTranslation(center.x, center.y, center.z);
    tet.applyMatrix(mat);

    var pointLight, pointLightHelper;
    for (i = 0; i < tet.vertices.length; i++) {
        pointLight = new THREE.PointLight(0xffffff, 0.5, 700);
        scene.add(pointLight);
        pointLight.position.copy(tet.vertices[i]);
        // pointLightHelper = new THREE.PointLightHelper( pointLight, 10 );
        // scene.add( pointLightHelper );
    }

    // add a small amount of background ambient light
    var ambientLight = new THREE.AmbientLight(0x444444);
    scene.add(ambientLight);    

    // reset button
    three_container.getElementsByClassName("reset_button")[0].addEventListener("click", function () { controls.reset(); });

    // checkbox for wireframe
    wf = three_container.getElementsByClassName("wireframe_checkbox")[0];
    wf.checked = material.wireframe;
    wf.addEventListener("change", function () { material.wireframe = wf.checked; });

    function animate() {
        if ( camera instanceof THREE.Camera === false || ! document.body.contains(three_container)) {
            console.log("Animation loop failed: stopping");
            return;
        }
        renderer.render(scene, camera);
        controls.update();
        requestAnimationFrame(animate);
    }
    animate();
}
