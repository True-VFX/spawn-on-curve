from functools import partial
from math import degrees

import numpy as np
import omni.ext
import omni.kit.commands
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

try:
    from .modules import bezier
except Exception:
    import omni.kit
    print("Attempting install of bezier")
    omni.kit.pipapi.install('bezier', ignore_import_check=True)

class Curve:
    def __init__(self, points:"list[list[float,float,float]]") -> None:
        # if not isinstance(points[0], (tuple, list)):
        #     points = [tuple(p) for pt in points for p in pt]
        self.points = points
        self.curves = self.calc_curves()
        self.length = sum(length for length, curve in self.curves)
    
    def calc_curves(self) -> "list[tuple[float,bezier.Curve]]":
        pts_per_curve = []
        curve_pts = []
        for i, pt in enumerate(self.points):
            curve_pts.append(pt)
            if i>1 and i % 3==0:
                pts_per_curve.append(curve_pts)
                curve_pts = [pt]
        

        # for i, crv in enumerate(pts_per_curve):
        #     print(f"Curve {i}")
        #     for pt in crv:
        #         print("    -", pt)
        
        curves = []
        for crv_pts in pts_per_curve:
            points = np.asfortranarray(crv_pts).T
            crv = bezier.Curve.from_nodes(points)
            curves.append((crv.length, crv))
        
        # print("Curves Found:", len(curves))
        
        return curves

    def evaluate(self, fac:float) -> "np.ndarray[float, float, float]":
        fac_length = fac * self.length
        
        crv, crv_ind = None, 0
        prev_length = 0
        running_len = 0
        for i, (length, _curve) in enumerate(self.curves):
            prev_length = running_len
            running_len += length
            if running_len > fac_length:
                crv_ind = i
                crv = self.curves[crv_ind][1]
                break

        # Reconfigure fac
        crv_start_fac = prev_length / self.length
        crv_end_fac = running_len / self.length
        recon_fac = (fac - crv_start_fac) / (crv_end_fac - crv_start_fac)

        return crv.evaluate(recon_fac)

    def evaluate_multi(self, *facs:"list[float]") -> np.ndarray:
        """
        Returns:
            numpy.ndarray: The points on the curve. As a two dimensional NumPy array, with the columns corresponding to each s value and the rows to the dimension.
        """
        locs = []
        for fac in facs[0]:
            loc = self.evaluate(fac).flatten().tolist()
            locs.append(Gf.Vec3d(loc[0], loc[1], loc[2]))

        return locs
    
    def eval_and_tans(self, fac:float) -> "tuple[np.ndarray, np.ndarray]":
        """Returns:
            position vector, tangent vector"""
        fac_length = fac * self.length
        
        crv, crv_ind = None, 0
        prev_length = 0
        running_len = 0
        for i, (length, _curve) in enumerate(self.curves):
            prev_length = running_len
            running_len += length
            if running_len >= fac_length:
                crv_ind = i
                crv = self.curves[crv_ind][1]
                break

        # Reconfigure fac
        crv_start_fac = prev_length / self.length
        crv_end_fac = running_len / self.length
        recon_fac = (fac - crv_start_fac) / (crv_end_fac - crv_start_fac)

        return crv.evaluate(recon_fac), crv.evaluate_hodograph(recon_fac)

    def eval_and_trans_multi(self, *facs:"list[float]") -> "tuple[np.ndarray, np.ndarray]":
        """
        Returns:
            positions: numpy.ndarray: The points on the curve. As a two dimensional NumPy array, with the columns corresponding to each s value and the rows to the dimension.
            tangents: numpy.ndarray
        """
        locs, tans = [], []
        locs_and_trans = [self.eval_and_tans(fac) for fac in facs[0]]
        for loc, tan in locs_and_trans:
            loc = loc.flatten().tolist()
            locs.append(Gf.Vec3d(loc[0], loc[1], loc[2]))
            tan = tan.flatten().tolist()
            tans.append(Gf.Vec3d(tan[0], tan[1], tan[2]))

        return locs, tans



# class SpawnOnCurve(omni.ext.IExt):
class _Ignore:
    def get_active_prim(self) -> Usd.Prim:
        s = omni.usd.get_context()
        prim_paths = s.get_selection().get_selected_prim_paths()
        if not prim_paths:
            return
        stage = s.get_stage()
        return stage.GetPrimAtPath(prim_paths[0])
    
    def get_stage(self) -> Usd.Stage:
        return omni.usd.get_context().get_stage()

    def get_beziers(self, curve_prim:Usd.Prim) -> Curve:
        pts = curve_prim.GetAttribute("points").Get()
        return Curve(pts)

    def get_bezier_from_basiscurve(self, curve_prim:Usd.Prim) -> bezier.Curve:
        b = curve_prim.GetAttribute("points")
        points = np.asfortranarray(b.Get()).T
        return bezier.Curve.from_nodes(np.array(points))

    def create_spheres_at_curve_points(self):
        s = omni.usd.get_context()
        stage:Usd.Stage = s.get_stage()
        curve_prim = self.get_active_prim()

        c_points = curve_prim.GetAttribute("points").Get()
        
        omni.kit.commands.execute("DeletePrims", paths=test_spheres)
        stage = self.get_stage()
        session_layer = stage.GetRootLayer()
        with Usd.EditContext(stage, session_layer):
            with Sdf.ChangeBlock():
                for i,pt in enumerate(c_points):
                    root_path = stage.GetDefaultPrim().GetPath()
                    sphere_path = root_path.AppendChild(f"Sphere_{i}")
                    sphere = session_layer.GetPrimAtPath(sphere_path) or Sdf.PrimSpec(session_layer.GetPrimAtPath(root_path), f"Sphere_{i}", Sdf.SpecifierDef, "Sphere")
                    pos = session_layer.GetAttributeAtPath(f"{sphere_path}.xformOp:translate") or Sdf.AttributeSpec(sphere, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                    pos.default = Gf.Vec3d(pt)
                    radius = session_layer.GetAttributeAtPath(f"{sphere_path}.radius") or Sdf.AttributeSpec(sphere, "radius", Sdf.ValueTypeNames.Double)
                    radius.default = 10
                    op_order = session_layer.GetAttributeAtPath(f"{sphere_path}.xformOpOrder") or Sdf.AttributeSpec(sphere, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    op_order.default  = ["xformOp:translate"]
                    test_spheres.append(sphere.path)

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
    
    def tangent_to_angle(self, tan):
        base = [1, 0, 0]
        tan = np.array(tan).flatten()
        unit_tan = tan / np.linalg.norm(tan)
        unit_base = base / np.linalg.norm(base)
        dot_product = np.dot(unit_tan, unit_base)
        return np.arccos(dot_product)

    def tanget_to_quaternion(self, tan) -> Gf.Quath:
        angle = self.tangent_to_angle(tan)
        quat = self.get_quaternion_from_euler(0,0,angle)
        return Gf.Quath(quat[0], quat[2], quat[1], quat[3])

    def move_cube_along_curve(self, slider:ui.UIntSlider,_b:float):
        s = omni.usd.get_context()
        stage:Usd.Stage = s.get_stage()
        curve_prim = self.get_active_prim()
        
        # Ensure Curve
        if not curve_prim or curve_prim.GetTypeName() != "BasisCurves":
            return

        # curve = self.get_bezier_from_basiscurve(curve_prim)
        mcurve = self.get_beziers(curve_prim)

        cube_prim = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendPath("Curve_Cube"))

        if not cube_prim:
            return

        # Move Cube
        fac = slider.model.get_value_as_float()
        # loc = mcurve.evaluate(fac).flatten().tolist()
        loc, tan = mcurve.eval_and_tans(fac)
        angle = degrees(self.tangent_to_angle(tan))
        if np.isnan(angle):
            angle = 0
        print("Angle:", angle)
        # quat = self.tanget_to_quaternion(tan)
        # orig_loc = curve.evaluate(fac).flatten().tolist()
        UsdGeom.XformCommonAPI(cube_prim).SetTranslate(loc.flatten().tolist())
        UsdGeom.XformCommonAPI(cube_prim).SetRotate((0,-angle,0))

    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        # Get Selected
        self._window = ui.Window("Spawn Along Curve", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                slider = ui.FloatSlider(min=0.0, max=1.0)
                slider.model.add_value_changed_fn(partial(self.move_cube_along_curve, slider))

                def on_click(slider:ui.FloatSlider):
                    curve_prim = self.get_active_prim()

                    # Ensure Curve
                    if curve_prim.GetTypeName() != "BasisCurves":
                        return
                    
                    curve = self.get_bezier_from_basiscurve(curve_prim)

                    # Spawn cube
                    C:omni.usd.UsdContext = omni.usd.get_context()
                    stage:Usd.Stage = C.get_stage()
                    cube_prim: UsdGeom.Cube = UsdGeom.Cube.Define(stage, stage.GetDefaultPrim().GetPath().AppendPath("Curve_Cube"))
                    # cube_prim: UsdGeom.Cube = UsdGeom.Cube.Define(stage, curve_prim.GetPath().AppendPath("Curve_Cube"))
                    cube_prim.GetSizeAttr().Set(100.0)
                    loc = curve.evaluate(slider.model.get_value_as_float()).flatten().tolist()
                    UsdGeom.XformCommonAPI(cube_prim).SetTranslate(loc)

                ui.Button("Spawn Along Curve", clicked_fn=lambda: on_click(slider), height=20)

                ui.Button("Spheres on Points", clicked_fn=self.create_spheres_at_curve_points, height=20)

    def on_shutdown(self):
        print("[tvfx.curve.spawnOnCurve] MyExtension shutdown")

test_spheres = []
# class _Ignore:
class SpawnOnCurve(omni.ext.IExt):
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
    
    def get_stage(self) -> Usd.Stage:
        return omni.usd.get_context().get_stage()

    def get_active_prim(self) -> Usd.Prim:
        s = omni.usd.get_context()
        prim_paths = s.get_selection().get_selected_prim_paths()
        if not prim_paths:
            return
        stage = s.get_stage()
        return stage.GetPrimAtPath(prim_paths[0])
    
    def get_beziers(self, curve_prim:Usd.Prim, is_closed:bool) -> Curve:
        pts = curve_prim.GetAttribute("points").Get()
        # print('pts:', tuple(pts[0]),",",type(tuple(pts[0])))
        pts = [tuple(pt) for pt in pts]
        # print("Point Count Pre:", len(pts))
        # if is_closed or curve_prim.GetAttribute("wrap").Get() == "nonperiodic":
        #     print("IS CLOSED")
        #     pts.append(pts[0])
        #     pts.append(pts[0])
        # print("Point Count Post:", len(pts), pts)
        return Curve(pts)

    def spawn_along_curve(self, count_slider:ui.UIntSlider, dist_slider:ui.FloatSlider, spawn_at_end:ui.CheckBox, is_closed_checkbox:ui.CheckBox, spawn_type:ui.ComboBox, spawn_obj_path:ui.StringField, _b:float):
        curve_prim = self.get_active_prim()

        curve_prim.GetAttribute("visibility").Set("inherited")

        # Ensure Curve
        if not curve_prim or curve_prim.GetTypeName() != "BasisCurves":
            if curve_prim:
                print(f"Active object must be a 'BasisCurve' not '{curve_prim.GetTypeName()}'")
            else:
                print("No active object")
            return

        p_inst = self.setup_stage(curve_prim, spawn_obj_path.model.get_value_as_string())

        # Ensure Curve objects will be seen
        curve_prim.GetAttribute("purpose").Set("default")


        is_closed = is_closed_checkbox.model.get_value_as_bool() or curve_prim.GetAttribute("wrap").Get() == "nonperiodic"
        curve = self.get_beziers(curve_prim, is_closed)

        # Spawn cube
        act_spawn_type = self.get_active_combo(spawn_type.model)
        if act_spawn_type == "Count":
            positions, ids, tangents = self.get_data_by_count(count_slider.model.get_value_as_int(),curve, spawn_at_end.model.get_value_as_bool())
        else:
            positions, ids, tangents = self.get_data_by_dist(dist_slider.model.get_value_as_float(), curve)

        quats = [
            self.tanget_to_quaternion(tan)
            for tan in tangents
        ]

        p_inst.CreateProtoIndicesAttr()
        p_inst.CreatePositionsAttr()
        p_inst.CreateOrientationsAttr()
        p_inst.GetProtoIndicesAttr().Set(ids)
        p_inst.GetPositionsAttr().Set(positions)
        p_inst.GetOrientationsAttr().Set(quats)
        # print(f"Made {len(ids)} cubes")

    def setup_stage(self, curve_prim:Usd.Prim, spawn_obj_path:str) -> UsdGeom.PointInstancer:
        stage = self.get_stage()
        p_inst:UsdGeom.PointInstancer = UsdGeom.PointInstancer.Define(stage, curve_prim.GetPath().AppendPath("Array_Holder"))
        p_inst.GetPrototypesRel().ClearTargets(True)

        if spawn_obj_path:
            spawn_prim = stage.GetPrimAtPath(spawn_obj_path)
        else:
            spawn_prim = UsdGeom.Cube.Define(stage,stage.GetDefaultPrim().GetPath().AppendChild("Cube"))
            spawn_prim.CreateSizeAttr(100)

        p_inst.CreatePrototypesRel()
        p_inst.GetPrototypesRel().AddTarget(spawn_prim.GetPath())

        return p_inst

    def get_data_by_count(self, count:int, curve:Curve, spawn_at_end:bool):
        if spawn_at_end:
            facs = np.array([0], dtype=np.double) if count == 1 else np.arange(count) / (count-1)
        else:
            facs = np.array([0], dtype=np.double) if count == 1 else np.arange(count) / count
        positions, tangents = curve.eval_and_trans_multi(facs)
        ids = [0] * count
        return positions, ids, tangents
        
    def get_data_by_dist(self, dist:float, curve:Curve):
        dist = max(0.00001, dist)
        curv_len = curve.length*0.01
        n = int(curv_len / dist) + 1

        facs = np.full(n, dist) * np.arange(n) / curv_len

        positions, tangents = curve.eval_and_trans_multi(facs)
        ids = [0] * n
        return positions, ids, tangents

    def tanget_to_quaternion(self, tan) -> Gf.Quath:
        # tan = tan.flatten()
        base = [1, 0, 0]
        unit_tan = tan / np.linalg.norm(tan)
        unit_base = base / np.linalg.norm(base)
        dot_product = np.dot(unit_tan, unit_base)
        angle = np.arccos(dot_product)
        quat = self.get_quaternion_from_euler(0, -angle, 0)
        return Gf.Quath(quat[0], quat[1], quat[2], quat[3])
        # return Gf.Quath(quat[0], quat[2], quat[1], quat[3])
    def get_active_combo(self, combo_model):
        return [
            combo_model.get_item_value_model(child).as_string
            for child in combo_model.get_item_children()
        ][combo_model.get_item_value_model().as_int]
    
    def drop_accept(self, _item):
        return True

    def drop(self, widget:ui.StringField, event:ui.WidgetMouseDropEvent):
        widget.model.set_value(event.mime_data)

    def on_startup(self, _ext_id):
        # Get Selected
        self._window = ui.Window("Spawn Along Curve", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                deflt = 0
                spawn_type = ui.ComboBox(deflt, "Count", "Distance")
                # act_spawn_type = self.get_active_combo(spawn_type.model)
                count_stack = ui.VStack(visible=deflt == 0)
                direction_stack = ui.VStack(visible=deflt == 1)
                
                with count_stack:
                    with ui.HStack():
                        ui.Label("Spawn Count:", height=0,width=90)
                        count_slider = ui.UIntSlider(min=0, max=100, height=0, tooltip="The number of items to spawn along the curve.")
                        count_slider.model.set_value(4)

                    with ui.HStack():
                        ui.Label("Spawn at End:", height=0,width=90)
                        spawn_at_end = ui.CheckBox(
                            height=20,
                            name="spawn_at_end",
                            tooltip="Whether or not to spawn at the very end of the curve (like the curve is a loop and you don't want it to spawn 2 in the same spot"
                        )
                        spawn_at_end.model.set_value(True)
                with direction_stack:
                    with ui.HStack():
                        ui.Label("Spawn Distance:", height=20,width=90)
                        dist_field = ui.FloatField(height=15, width=50)
                        dist_slider = ui.FloatSlider(
                            min=0.1, max=10.0, height=0, step=0.1,
                            tooltop="Spawn items every n distance along curve",
                            model=dist_field.model
                        )
                        dist_slider.model.set_value(0.75)

                with ui.HStack(visible=False):
                    ui.Label("Treat as Closed:", height=0,width=100)
                    treat_as_closed = ui.CheckBox(
                        height=20,
                        name="is_closed",
                        tooltip="Whether or not to treat an open curve as a closed loop, ie no discernable beginning or end points. If curve is already closed this check box does nothing and the curve is treated as closed"
                    )

                with ui.HStack():
                    ui.Label("Spawn Object:", height=20,width=100)
                    spawn_obj_path = ui.StringField(height=20)
                    spawn_obj_path.set_accept_drop_fn(self.drop_accept)
                    spawn_obj_path.set_drop_fn(lambda a, w=spawn_obj_path: self.drop(w, a))
                
                ui.Button("Spawn Along Curve", clicked_fn=lambda: self.spawn_along_curve(count_slider, dist_slider, spawn_at_end, treat_as_closed, spawn_type, spawn_obj_path, 0))

                def spawn_type_changed(combo_model, _item):
                    act_opt = self.get_active_combo(combo_model)
                    count_stack.visible = act_opt == 'Count'
                    direction_stack.visible = act_opt == 'Distance'
                    self.spawn_along_curve(count_slider, dist_slider, spawn_at_end, treat_as_closed, spawn_type, spawn_obj_path, 0)

                spawn_type.model.add_item_changed_fn(spawn_type_changed)

                changed_func = partial(self.spawn_along_curve, count_slider, dist_slider, spawn_at_end, treat_as_closed, spawn_type, spawn_obj_path)
                spawn_at_end.model.add_value_changed_fn(changed_func)
                count_slider.model.add_value_changed_fn(changed_func)
                dist_slider.model.add_value_changed_fn(changed_func)
                treat_as_closed.model.add_value_changed_fn(changed_func)
                spawn_obj_path.model.add_value_changed_fn(changed_func)

    def on_shutdown(self):
        print("[tvfx.curve.spawnOnCurve] MyExtension shutdown")
