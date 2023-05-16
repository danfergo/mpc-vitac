from yarok import component, interface


@component(
    tag='tumble-tower',
    defaults={
        'rows': 12,
        'block_width': 0.04,
        'block_height': 0.028,
        'block_length': 0.12
    },
    # language=xml
    template="""
        <mujoco>
            <asset>
                <texture 
                    name="wood_texture"
                    type="cube" 
                    file="wood.png"
                    width="2832" 
                    height="2832"/>
                <material name="wood" texture="wood_texture" rgba="0.8 0.8 0.8 1" specular="0.1"/>
            </asset>
            <default>
                <default class='tt-block'>
                    <geom type='box'
                        material='wood'
                        mass='0.05'
                        size="${block_width} ${block_length} ${block_height}"/>

                </default>
            </default>
            <worldbody>
                <for each='range(rows)' as='z'>
                    <for each='range(3)' as='x'>
                        <body pos="
                                ${0.5 + 0.082*x if z % 2 == 0 else 0.58} 
                                ${0.48 if z % 2 == 0 else 0.4 + 0.082*x}
                                ${0.061*z}" 
                            euler="0 0 ${0 if z % 2 == 0 else 1.57}">
                            <freejoint/>
                            <geom class='tt-block'/>
                        </body>
                    </for>
                </for>
            </worldbody>
        </mujoco>
    """
)
class TumbleTower:
    pass
