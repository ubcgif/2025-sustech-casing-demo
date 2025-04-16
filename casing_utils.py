import numpy as np
import discretize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

def get_theta_ind_mirror(mesh, theta_ind):
    return (
        theta_ind+int(mesh.vnC[1]/2)
        if theta_ind < int(mesh.vnC[1]/2)
        else theta_ind-int(mesh.vnC[1]/2)
    )

def mesh2d_from_3d(mesh):
        """
        create cylindrically symmetric mesh generator
        """
        mesh2D = discretize.CylindricalMesh(
            [mesh.h[0], 1., mesh.h[2]], x0=mesh.x0
        )
        return mesh2D

def plot_slice(
    mesh, v, ax=None, clim=None, pcolor_opts=None, theta_ind=0,
    cb_extend=None, show_cb=True
):
    """
    Plot a cell centered property

    :param numpy.array prop: cell centered property to plot
    :param matplotlib.axes ax: axis
    :param numpy.array clim: colorbar limits
    :param dict pcolor_opts: dictionary of pcolor options
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if pcolor_opts is None:
        pcolor_opts = {}
    if clim is not None:
        norm = Normalize(vmin=clim.min(), vmax=clim.max())
        pcolor_opts["norm"] = norm

    # generate a 2D mesh for plotting slices
    mesh2D = mesh2d_from_3d(mesh)

    vplt = v.reshape(mesh.vnC, order="F")
    plotme = discretize.utils.mkvc(vplt[:, theta_ind, :])
    if not mesh.is_symmetric:
        theta_ind_mirror = get_theta_ind_mirror(mesh, theta_ind)
        mirror_data = discretize.utils.mkvc(vplt[:, theta_ind_mirror, :])
    else:
        mirror_data = plotme

    out = mesh2D.plot_image(
        plotme, ax=ax,
        mirror=True, mirror_data=mirror_data,
        pcolor_opts=pcolor_opts,
    )

    out += (ax, )

    if show_cb:
        cb = plt.colorbar(
            out[0], ax=ax,
            extend=cb_extend if cb_extend is not None else "neither"
        )
        out += (cb, )

        # if clim is not None:
        #     cb.set_clim(clim)
        #     cb.update_ticks()

    return out

def plotFace2D(
    mesh2D,
    j, real_or_imag="real", ax=None, range_x=None,
    range_y=None, sample_grid=None,
    log_scale=True, clim=None, mirror=False, mirror_data=None,
    pcolor_opts=None,
    show_cb=True,
    stream_threshold=None, stream_opts=None
):
    """
    Create a streamplot (a slice in the theta direction) of a face vector

    :param discretize.CylMesh mesh2D: cylindrically symmetric mesh
    :param np.ndarray j: face vector (x, z components)
    :param str real_or_imag: real or imaginary component
    :param matplotlib.axes ax: axes
    :param numpy.ndarray range_x: x-extent over which we want to plot
    :param numpy.ndarray range_y: y-extent over which we want to plot
    :param numpy.ndarray sample_grid: x, y spacings at which to re-sample the plotting grid
    :param bool log_scale: use a log scale for the colorbar?
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(j) == mesh2D.nF:
        vType = "F"
    elif len(j) == mesh2D.nC*2:
        vType = "CCv"

    if pcolor_opts is None:
        pcolor_opts = {}

    if log_scale is True:
        if clim is not None:
            pcolor_opts["norm"] = LogNorm(vmin=clim.min(), vmax=clim.max())
        else:
            pcolor_opts["norm"] = LogNorm()
    else:
        if clim is not None:
            norm = Normalize(vmin=clim.min(), vmax=clim.max())
            pcolor_opts["norm"] = norm



    f = mesh2D.plot_image(
        getattr(j, real_or_imag),
        view="vec", vType=vType, ax=ax,
        range_x=range_x, range_y=range_y, sample_grid=sample_grid,
        mirror=mirror,
        mirror_data= (
            getattr(mirror_data, real_or_imag) if mirror_data is not None
            else None)
        ,
        pcolor_opts=pcolor_opts, stream_threshold=stream_threshold,
        streamOpts=stream_opts
    )

    out = f + (ax,)

    if show_cb is True:
        cb = plt.colorbar(f[0], ax=ax)
        out += (cb,)

        # if clim is not None:
        #     cb.set_clim(clim)
        #     cb.update_ticks()

    return out


def plotEdge2D(
    mesh2D,
    h, real_or_imag="real", ax=None, range_x=None,
    range_y=None, sample_grid=None,
    log_scale=True, clim=None, mirror=False, pcolor_opts=None
):
    """
    Create a pcolor plot (a slice in the theta direction) of an edge vector

    :param discretize.CylMesh mesh2D: cylindrically symmetric mesh
    :param np.ndarray h: edge vector (y components)
    :param str real_or_imag: real or imaginary component
    :param matplotlib.axes ax: axes
    :param numpy.ndarray range_x: x-extent over which we want to plot
    :param numpy.ndarray range_y: y-extent over which we want to plot
    :param numpy.ndarray sample_grid: x, y spacings at which to re-sample the plotting grid
    :param bool log_scale: use a log scale for the colorbar?
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(h) == mesh2D.nE:
        vType = "E"
    elif len(h) == mesh2D.nC:
        vType = "CC"
    elif len(h) == 2*mesh2D.nC:
        vType = "CCv"

    if log_scale is True:
        pcolor_opts["norm"] = LogNorm()
    else:
        pcolor_opts = {}

    cb = plt.colorbar(
        mesh2D.plot_image(
            getattr(h, real_or_imag),
            view="real", vType=vType, ax=ax,
            range_x=range_x, range_y=range_y, sample_grid=sample_grid,
            mirror=mirror,
            pcolor_opts=pcolor_opts,
        )[0], ax=ax
    )

    if clim is not None:
        cb.set_clim(clim)

    return ax, cb

def plot_cross_section(
    fields,
    view=None,
    ax=None,
    xlim=None,
    zlim=None,
    clim=None,
    prim_sec=None,
    primary_fields=None,
    casing_a=None,
    casing_b=None,
    casing_z=None,
    real_or_imag="real",
    theta_ind=0,
    src_ind=0,
    time_ind=0,
    casing_outline=False,
    cb_extend=None,
    show_cb=True,
    show_mesh=False,
    use_aspect=False,
    stream_opts=None,
    log_scale=True,
    epsilon=1e-20
):
    """
    Plot the fields
    """

    # create default at
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    mesh = fields.simulation.mesh
    physics = fields.__module__.split(".")[-2]

    # 2D mesh for plotting
    if mesh.is_symmetric:
        mesh2D = mesh
    else:
        mesh2D = mesh2d_from_3d(mesh)

    plotme = get_plotting_data(
        fields, view=view, src_ind=src_ind, time_ind=time_ind,
        prim_sec=prim_sec, primary_fields=primary_fields,
    )

    if not mesh.is_symmetric:
        theta_ind_mirror = get_theta_ind_mirror(mesh, theta_ind)
    else:
        theta_ind_mirror = 0
        mirror_data=None

    norm = None
    if view in ["charge", "charge_density", "phi", "sigma", "mur", "mu"]:
        plot_type = "scalar"
        if view == "sigma":
            norm = LogNorm()
        elif view in ["charge", "charge_density"]:
            if clim is None:
                clim = np.r_[-1., 1.] * np.max(np.absolute(plotme))
            else:
                clim = np.r_[-1., 1.] * np.max(np.absolute(clim))

        if view == "phi" and log_scale is True:
            norm = SymLogNorm(
                clim[0] if clim is not None else
                np.max([epsilon, np.min(np.absolute(plotme))])
            )
            clim = clim[1]*np.r_[-1., 1.] if clim is not None else None

        # if not mesh.is_symmetric:
        plotme = plotme.reshape(mesh.vnC, order="F")
        if physics == "frequency_domain":
            plotme = plotme[:, :, None]
        mirror_data = discretize.utils.mkvc(
            plotme[:, theta_ind_mirror, :]
        )
        plotme = discretize.utils.mkvc(plotme[:, theta_ind, :])


    else:
        if len(plotme) == np.sum(mesh.vnF):
            plt_vec = face3DthetaSlice(mesh, plotme, theta_ind=theta_ind)
            mirror_data = face3DthetaSlice(mesh, plotme, theta_ind=theta_ind_mirror)
            plot_type = "vec"

        else:

            plot_type = "scalar"

            if len(mesh.h[1]) == 1:
                plotme = mesh.aveE2CC * plotme

            else:
                plotme = (mesh.aveE2CCV * plotme)[mesh.nC:2*mesh.nC]

            plotme = plotme.reshape(mesh.vnC, order="F")
            mirror_data = discretize.utils.mkvc(-plotme[:, theta_ind_mirror, :])
            plotme = discretize.utils.mkvc(plotme[:, theta_ind, :])

            norm = SymLogNorm(
                clim[0] if clim is not None else
                np.max([epsilon, np.min(np.absolute(plotme))])
            )
            clim = clim[1]*np.r_[-1., 1.] if clim is not None else None

    if plot_type == "scalar":
        out = mesh2D.plot_image(
            getattr(plotme, real_or_imag), ax=ax,
            pcolor_opts = {
                "cmap": "RdBu_r" if view in ["charge", "charge_density"] else "viridis",
                "norm": norm
            },
            mirror_data=mirror_data,
            mirror=True
        )

        if show_cb:
            cb = plt.colorbar(
                out[0], ax=ax,
                extend="neither" if cb_extend is None else cb_extend,
            )

            out += (cb,)

    elif plot_type == "vec":
        out = plotFace2D(
            mesh2D,
            plt_vec + epsilon,
            real_or_imag=real_or_imag,
            ax=ax,
            range_x=xlim,
            range_y=zlim,
            sample_grid=(
                np.r_[np.diff(xlim)/100., np.diff(zlim)/100.]
                if xlim is not None and zlim is not None else None
            ),
            log_scale=True,
            clim=clim,
            stream_threshold=clim[0] if clim is not None else None,
            mirror=True,
            mirror_data=mirror_data,
            stream_opts=stream_opts,
            show_cb=show_cb,
        )


    if show_mesh is True:
        mesh2D.plot_grid(ax=ax)

    title = "{} {}".format(prim_sec, view)
    if physics == "frequency_domain":
        title += "\nf = {:1.1e} Hz".format(fields.survey.source_list[src_ind].frequency)
    elif physics == "time_domain":
        title += "\n t = {:1.1e} s".format(
            fields._times[time_ind]
        )
    ax.set_title(
        title, fontsize=13
    )
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)

    # plot outline of casing
    if casing_outline is True:
        factor = [-1, 1]
        [
            ax.plot(
                fact * np.r_[
                    casing_a, casing_a, casing_b,
                    casing_b, casing_a
                ],
                np.r_[
                    casing_z[1], casing_z[0], casing_z[0],
                    casing_z[1], casing_z[1]
                ],
                "k",
                lw = 0.5
            )
            for fact in factor
        ]

    if use_aspect is True:
        ax.set_aspect(1)

    return out
