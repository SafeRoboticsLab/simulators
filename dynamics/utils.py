import pybullet as p

def change_dynamics(body_id, link_id=-1, mass=None, lateral_friction=None, spinning_friction=None,
                        rolling_friction=None, anisotropic_friction=None, restitution=None, linear_damping=None, angular_damping=None,
                        contact_stiffness=None, contact_damping=None, friction_anchor=None,
                        local_inertia_diagonal=None, inertia_position=None, inertia_orientation=None,
                        joint_damping=None, joint_friction=None, joint_force=None):

    kwargs = {}
    if mass is not None:
        kwargs['mass'] = mass
    if lateral_friction is not None:
        kwargs['lateralFriction'] = lateral_friction
    if spinning_friction is not None:
        kwargs['spinningFriction'] = spinning_friction
    if rolling_friction is not None:
        kwargs['rollingFriction'] = rolling_friction
    if anisotropic_friction is not None:
        kwargs['anisotropicFriction'] = anisotropic_friction
    if restitution is not None:
        kwargs['restitution'] = restitution
    if linear_damping is not None:
        kwargs['linearDamping'] = linear_damping
    if angular_damping is not None:
        kwargs['angularDamping'] = angular_damping
    if contact_stiffness is not None:
        kwargs['contactStiffness'] = contact_stiffness
    if contact_damping is not None:
        kwargs['contactDamping'] = contact_damping
    if friction_anchor is not None:
        kwargs['frictionAnchor'] = friction_anchor
    if local_inertia_diagonal is not None:
        kwargs['localInertiaDiagonal'] = local_inertia_diagonal
    if joint_damping is not None:
        kwargs['jointDamping'] = joint_damping
    if joint_force is not None:
        kwargs['jointLimitForce'] = joint_force

    p.changeDynamics(body_id, link_id, **kwargs)