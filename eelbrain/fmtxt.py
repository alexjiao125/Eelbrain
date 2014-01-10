"""
Text objects which can be output as str but also as formatted text (currently
TeX and HTML). Different classes can be used to achieve special formatting:

:class:`FMText`
    Any string of text, along with a formatting property specified as TeX
    property. FMText objects can be sequenced and nested.
:class:`Table`
    Tables with multicolumn cells.
:class:`Image`
    Images.
:class:`Figure`
    Figure with content and caption.
:class:`Section`
    Document section containing a title and content (any other FMText objects,
    including other Section objects for subsections).
:class:`Report`
    Document consisting of several sections plus a title.

Whenever an parameter asks for an FMText object, a plain str can also be
provided and will be converted to an FMText object.

FMText objects provide an interface to formatting through different methods:

- the :py:meth:`__str__` method for a string representation
- a :py:meth:`get_tex` method for a TeX representation
- a :py:meth:`get_html` method for a HTML representation

The module also provides functions that work with fmtxt objects:

- :func:`save_tex` for saving an object's tex representation
- :func:`copy_tex` for copying an object's tex representation to
  the clipboard
- :func:`save_pdf` for saving a pdf
- :func:`copy_pdf` for copying a pdf to the clipboard
- :func:`save_html` for saving an HTML file

"""
# Author:  Christian Brodbeck <christianbrodbeck@nyu.edu>


import datetime
import logging
import os
import cPickle as pickle
import shutil
from StringIO import StringIO
import tempfile

try:
    import tex
except:
    logging.warning("module tex not found; pdf export not available")
    tex = None

import numpy as np

from . import ui


preferences = dict(
                   keep_recent=3,  # number of recent tables to keep in memory
                   html_tables_in_fig=True,
                   )

_html_alignments = {'l': 'left',
                    'r': 'right',
                    'c': 'center'}

_html_tags = {r'_': 'sub',
              r'^': 'sup',
              r'\author': 'author',
              r'\emph': 'em',
              r'\textbf': 'b',
              r'\textit': 'i'}

_html_doc_template = """<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
</head>

<body>

{body}

</body>
</html>
"""

# to keep track of recent tex out and allow copying
_recent_texout = []
def _add_to_recent(tex_obj):
    keep_recent = preferences['keep_recent']
    if keep_recent:
        if len(_recent_texout) >= keep_recent - 1:
            _recent_texout.pop(0)
        _recent_texout.append(tex_obj)


def isstr(obj):
    return isinstance(obj, basestring)


def get_pdf(tex_obj):
    "creates a pdf from an fmtxt object (using tex)"
    txt = tex_obj.get_tex()
    document = u"""
\\documentclass{article}
\\usepackage{booktabs}
\\begin{document}
%s
\\end{document}
""" % txt
    pdf = tex.latex2pdf(document)
    return pdf


def save_html(fmtxt, path=None):
    "Save an FMText object as html file"
    if path is None:
        msg = "Save as HTML"
        path = ui.ask_saveas(msg, msg, [('HTML (*.html)', '*.html')])
        if not path:
            return
    path = os.path.abspath(path)

    extension = '.html'
    if path.endswith(extension):
        resource_dir = path[:-len(extension)]
        file_path = path
    else:
        resource_dir = path
        file_path = path + extension

    if os.path.exists(resource_dir):
        shutil.rmtree(resource_dir)
    os.mkdir(resource_dir)

    root = os.path.dirname(file_path)
    resource_dir = os.path.relpath(resource_dir, root)

    buf = make_html_doc(fmtxt, root, resource_dir)
    with open(file_path, 'wb') as fid:
        fid.write(buf)


def save_pdf(tex_obj, path=None):
    "Save an fmtxt object as a pdf"
    pdf = get_pdf(tex_obj)
    if path is None:
        msg = "Save as PDF"
        path = ui.ask_saveas(msg, msg, [('PDF (*.pdf)', '*.pdf')])
    if path:
        with open(path, 'w') as f:
            f.write(pdf)


def save_tex(tex_obj, path=None):
    "saves an fmtxt object as a pdf"
    txt = tex_obj.get_tex()
    if path is None:
        path = ui.ask_saveas(title="Save tex", ext=[('tex', 'tex source code')])
    if path:
        with open(path, 'w') as f:
            f.write(txt)


def copy_pdf(tex_obj=-1):
    """
    copies an fmtxt object to the clipboard as pdf. `tex_obj` can be an object
    with a `.get_tex` method or an int, in which case the item is retrieved from
    a list of recently displayed fmtxt objects.

    """
    if isinstance(tex_obj, int):
        tex_obj = _recent_texout[tex_obj]

    # save pdf to temp file
    pdf = get_pdf(tex_obj)
    fd, path = tempfile.mkstemp('.pdf', text=True)
    os.write(fd, pdf)
    os.close(fd)
    logging.debug("Temporary file created at: %s" % path)

    # copy to clip-board
    ui.copy_file(path)


def copy_tex(tex_obj):
    "copies an fmtxt object to the clipboard as tex code"
    txt = tex_obj.get_tex()
    ui.copy_text(txt)


def html(text, options={}):
    """Create html code for any object with a string representation

    Parameters
    ----------
    text : any
        Object to be converted to HTML. If the object has a ``.get_html()``
        method the result of this method is returned, otherwise ``str(text)``.

    Options
    -------
    ...
    """
    if hasattr(text, 'get_html'):
        return text.get_html(options)
    else:
        return str(text)


def make_html_doc(body, root, resource_dir, title=None):
    """
    Parameters
    ----------
    body : fmtxt-object
        FMTXT object which should be formatted into an HTML document.
    root : str
        Path to the directory in which the HTML file is going to be located.
    resource_dir : str
        Path to the directory containing resources like images, relative to
        root.
    title : None | FMText
        Document title. Default is title specified by body.get_title() or
        "Untitled".

    Returns
    -------
    html : str
        HTML document.
    """
    if title is None:
        if hasattr(body, 'get_title'):
            title = html(body.get_title())
        else:
            title = "Untitled"

    options = {'root': root, 'resource_dir': resource_dir}
    txt_body = html(body, options)
    txt = _html_doc_template.format(title=title, body=txt_body)
    return txt


def texify(txt):
    """
    prepares non-latex txt for input to tex (e.g. for Matplotlib)

    """
    if hasattr(txt, 'get_tex'):
        return txt.get_tex()
    elif not isstr(txt):
        txt = str(txt)

    out = txt.replace('_', r'\_') \
             .replace('{', r'\{') \
             .replace('}', r'\}')
    return out


_html_temp = '<{tag}>{body}</{tag}>'
_html_temp_opt = '<{tag} {options}>{body}</{tag}>'
def _html_element(tag, body, options=None):
    """Format an HTML element

    Parameters
    ----------
    tag : str
        The HTML tag.
    body : FMText
        The main content between the tags.
    options : dict
        Options to be inserted in the start tag.
    """
    if options:
        options_ = ' '.join('%s="%s"' % item for item in options.iteritems())
        txt = _html_temp_opt.format(tag=tag, options=options_, body=html(body))
    else:
        txt = _html_temp.format(tag=tag, body=html(body))
    return txt


def asfmtext(text):
    "Convert non-FMText objects to FMText"
    if isinstance(text, FMTextElement):
        return text
    elif np.iterable(text) and not isstr(text):
        return FMText(text)
    else:
        return FMTextElement(text)


class FMTextElement(object):
    "Represent a value along with formatting properties."
    def __init__(self, content, property=None, mat=False, drop0=False,
                 fmt='%.6g'):
        """Represent a value along with formatting properties.

        Parameters
        ----------
        content : object
            Any item with a string representation (str, scalar, ...).
        property : str
            TeX property that is followed by ``{}`` (e.g.,
            ``property=r'\textbf'`` for bold)
        mat : bool
            For TeX output, content is enclosed in ``'$...$'``
        drop0 : bool
            For  numbers smaller than 0, drop the '0' before the decimal
            point (e.g., for p values).
        fmt : str
            Format-str for numerical values.
        """
        self._content = content
        self.mat = mat
        self.drop0 = drop0
        self.fmt = fmt
        self.property = property

    def __repr__(self):
        name = self.__class__.__name__
        args = ', '.join(self.__repr_items__())
        return "%s(%s)" % (name, args)

    def __repr_items__(self):
        items = [repr(self._content)]
        if self.property:
            items.append(repr(self.property))
        if self.mat:
            items.append('mat=True')
        if self.drop0:
            items.append('drop0=True')
        if self.fmt != '%s':
            items.append('fmt=%r' % self.fmt)
        return items

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.get_str()

    def __add__(self, other):
        if isinstance(other, str) and other == '':
            # added to prevent matplotlib from thinking Image is a file path
            raise ValueError("Can't add empty string")

        return FMText([self, other])

    def get_html(self, options):
        txt = self._get_html_core(options)

        if self.property is not None and self.property in _html_tags:
            tag = _html_tags[self.property]
            txt = _html_element(tag, txt)

        return txt

    def _get_html_core(self, options):
        return self.get_str(options)

    def get_str(self, options={}):
        if isstr(self._content):
            return self._content.replace('\\', '')
        elif self._content is None:
            return ''
        elif np.isnan(self._content):
            return 'NaN'
        elif isinstance(self._content, (bool, np.bool_, np.bool8)):
            return '%s' % self._content
        elif np.isscalar(self._content) or getattr(self._content, 'ndim', None) == 0:
            fmt = options.get('fmt', self.fmt)
            txt = fmt % self._content
            if self.drop0 and len(txt) > 2 and txt.startswith('0.'):
                txt = txt[1:]
            return txt
        elif not self._content:
            return ''
        else:
            msg = "Unknown text in FMTextElement: %s" % str(self._content)
            logging.warning(msg)
            return ''

    def get_tex(self, options):
        txt = self._get_tex_core(options)

        if self.property:
            txt = r"%s{%s}" % (self.property, txt)

        if self.mat and not options.get('mat', False):
            txt = "$%s$" % txt

        return txt

    def _get_tex_core(self, options):
        return self.get_str(options)


class FMText(FMTextElement):
    """Represent a value along with formatting properties.

    The elementary unit of the :py:mod:`fmtxt` module. It can function as a
    string, but can hold formatting properties such as font properties.

    The following methods are used to get different string representations:

     - FMText.get_str() -> unicode
     - FMText.get_tex() -> str (TeX)
     - FMText.get_html() -> str (HTML)
     - str(FMText) -> str

    """
    def __init__(self, content, property=None, mat=False,
                 drop0=False, fmt='%.6g'):
        """Represent a value along with formatting properties.

        Parameters
        ----------
        content : str | object | iterable
            Any item with a string representation (str, FMText, scalar, ...)
            or an object that iterates over such items (e.g. a list of FMText).
        property : str
            TeX property that is followed by ``{}`` (e.g.,
            ``property=r'\textbf'`` for bold)
        mat : bool
            For TeX output, content is enclosed in ``'$...$'``
        drop0 : bool
            For  numbers smaller than 0, drop the '0' before the decimal
            point (e.g., for p values).
        fmt : str
            Format-str for numerical values.
        """
        # np integers are not instance of int
        if isinstance(content, np.integer):
            content = int(content)

        # convert to list of FMText
        if isinstance(content, FMTextElement):
            content = [content]
        elif (np.iterable(content) and not isstr(content)):  # lists
            content = map(asfmtext, content)
        else:
            content = [FMTextElement(content, None, mat, drop0, fmt)]

        FMTextElement.__init__(self, content, property, mat, drop0, fmt)

    def append(self, content):
        self._content.append(asfmtext(content))

    def _get_html_core(self, options):
        return ''.join(i.get_html(options) for i in self._content)

    def get_str(self, options={}):
        """
        Returns the string representation.

        Parameters
        ----------
        fmt : str
            can be used to override the format string associated with the
            texstr object
        """
        return ''.join(i.get_str(options) for i in self._content)

    def _get_tex_core(self, options):
        options_mat = options.get('mat', False)
        mat = self.mat or options_mat
        if mat != options_mat:
            options = options.copy()
            options['mat'] = mat
        return ''.join(i.get_tex(options) for i in self._content)


class symbol(FMTextElement):
    "Print df neatly in plain text as well as formatted"
    def __init__(self, symbol, df=None):
        assert (df is None) or np.isscalar(df) or isstr(df) or np.iterable(df)
        self._df = df
        FMTextElement.__init__(self, symbol, mat=True)

    def get_df_str(self):
        if np.isscalar(self._df):
            return '%s' % self._df
        elif isstr(self._df):
            return self._df
        else:
            return ','.join(str(i) for i in self._df)

    def _get_html_core(self, options):
        symbol = FMTextElement._get_html_core(self, options)
        if self._df is None:
            return symbol
        else:
            text = '%s<sub>%s<\sub>' % (symbol, self.get_df_str())
            return text

    def get_str(self, options={}):
        symbol = FMTextElement.get_str(self, options)
        if self._df is None:
            return symbol
        else:
            return '%s(%s)' % (symbol, self.get_df_str())

    def _get_tex_core(self, options):
        out = FMTextElement._get_tex_core(self, options)
        if self._df is not None:
            out += '_{%s}' % self.get_df_str()
        return out


def p(p, digits=3, stars=None, of=3):
    """Create an FMText representation of a p-value

    Parameters
    ----------
    p : scalar
        P-value.
    digits : int
        Significant digits.
    stars : None | str
        Stars decorating the p-value (e.g., "**")
    of : int
        Max numbers of star characters possible (to add empty space for
        alignment).

    Returns
    -------
    text : FMText
        FMText with formatted p-value.
    """
    if p < 10 ** -digits:  # APA 6th, p. 114
        p = '< .' + '0' * (digits - 1) + '1'
        mat = True
    else:
        mat = False
    fmt = '%' + '.%if' % digits
    text = FMText(p, fmt=fmt, drop0=True, mat=mat)
    if stars is not None:
        stars_text = Stars(stars, of=of)
        text.append(stars_text)
    return text


def stat(x, fmt="%.2f", stars=None, of=3, drop0=False):
    """
    returns a FMText with properties set for a statistic (e.g. a t-value)

    """
    ts_stat = FMText(x, fmt=fmt, drop0=drop0)
    if stars is None:
        return ts_stat
    else:
        ts_s = Stars(stars, of=of)
        return FMText((ts_stat, ts_s), mat=True)


def eq(name, result, eq='=', df=None, fmt='%.2f', drop0=False, stars=None,
       of=3):
    symbol_ = symbol(name, df=df)
    stat_ = stat(result, fmt=fmt, drop0=drop0, stars=stars, of=of)
    return FMText([symbol_, eq, stat_], mat=True)


class Stars(FMTextElement):
    """FMTextElement object for decoration of p-calues

    Shortcut for adding stars to a table and spaces in place of absent stars,
    so that alignment to the right can be used.
    """
    def __init__(self, n, of=3, property="^"):
        if isstr(n):
            self.n = len(n.strip())
        else:
            self.n = n
        self.of = of
        if np.isreal(n):
            text = '*' * n + ' ' * (of - n)
        else:
            text = n.ljust(of)
        FMTextElement.__init__(self, text, property, mat=True)

    def _get_tex_core(self, options):
        txt = str(self)
        spaces = r'\ ' * (self.of - self.n)
        return txt + spaces


# Table ---

class Cell(FMText):
    def __init__(self, text=None, property=None, width=1, just=None,
                 **texstr_kwargs):
        """A cell for a table

        Parameters
        ----------
        text : FMText
            Cell content.
        width : int
            Width in columns for multicolumn cells.
        just : None | 'l' | 'r' | 'c'
            Justification. None: use column standard.
        others :
            FMText parameters.
        """
        FMText.__init__(self, text, property, **texstr_kwargs)
        self.width = width
        if width > 1 and not just:
            self.just = 'l'
        else:
            self.just = just

    def __repr_items__(self):
        items = FMText.__repr_items__(self)
        if self.width != 1:
            i = min(2, len(items))
            items.insert(i, 'width=%s' % self.width)
        return items

    def __len__(self):
        return self.width

    def get_html(self, options={}):
        html_repr = FMText.get_html(self, options)
        options = []
        if self.width > 1:
            options.append('colspan="%i"' % self.width)
        if self.just:
            align = _html_alignments[self.just]
            options.append('align="%s"' % align)

        if options:
            start_tag = '<td %s>' % ' '.join(options)
        else:
            start_tag = '<td>'

        html_repr = ' %s%s</td>' % (start_tag, html_repr)
        return html_repr

    def get_tex(self, options={}):
        tex_repr = FMText.get_tex(self, options)
        if self.width > 1 or self.just:
            tex_repr = r"\multicolumn{%s}{%s}{%s}" % (self.width, self.just,
                                                      tex_repr)
        return tex_repr


class Row(list):
    def __len__(self):
        return sum([len(cell) for cell in self])

    def __repr__(self):
        return "Row(%s)" % list.__repr__(self)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return ' '.join([str(cell) for cell in self])

    def _strlen(self, options):
        "returns list of cell-str-lengths; multicolumns handled poorly"
        lens = []
        for cell in self:
            cell_len = len(cell.get_str(options))
            for _ in xrange(len(cell)):
                lens.append(cell_len / len(cell))  # TODO: better handling of multicolumn
        return lens

    def get_html(self, options={}):
        html = '\n'.join(cell.get_html(options) for cell in self)
        html = '<tr>\n%s\n</tr>' % html
        return html

    def get_str(self, c_width, c_just, delimiter='   ',
                options={}):
        "returns the row using col spacing provided in c_width"
        col = 0
        out = []
        for cell in self:
            if cell.width == 1:
                strlen = c_width[col]
                if cell.just:
                    just = cell.just
                else:
                    just = c_just[col]
            else:
                strlen = sum(c_width[col:col + cell.width])
                strlen += len(delimiter) * (cell.width - 1)
                just = cell.just
            col += cell.width
            txt = cell.get_str(options)
            if just == 'l':
                txt = txt.ljust(strlen)
            elif just == 'r':
                txt = txt.rjust(strlen)
            elif just == 'c':
                rj = strlen / 2
                txt = txt.rjust(rj).ljust(strlen)
            out.append(txt)
        return delimiter.join(out)

    def get_tex(self, options={}):
        tex = ' & '.join(cell.get_tex(options) for cell in self)
        tex += r" \\"
        return tex

    def get_tsv(self, delimiter, fmt=None):
        options = {'fmt': fmt}
        txt = delimiter.join(cell.get_str(options) for cell in self)
        return txt


class Table(FMTextElement):
    """
    A table that can be output in text with equal width font
    as well as tex.

    Example::

        >>> from eelbrain import fmtxt
        >>> table = fmtxt.Table('lll')
        >>> table.cell()
        >>> table.cell("example 1")
        >>> table.cell("example 2")
        >>> table.midrule()
        >>> table.cell("string")
        >>> table.cell('???')
        >>> table.cell('another string')
        >>> table.cell("Number")
        >>> table.cell(4.5)
        >>> table.cell(2./3, fmt='%.4g')
        >>> print table
                 example 1   example 2
        -----------------------------------
        string   ???         another string
        Number   4.5         0.6667
        >>> print table.get_tex()
        \begin{center}
        \begin{tabular}{lll}
        \toprule
         & example 1 & example 2 \\
        \midrule
        string & ??? & another string \\
        Number & 4.5 & 0.6667 \\
        \bottomrule
        \end{tabular}
        \end{center}
        >>> table.save_tex()


    """
    def __init__(self, columns, rules=True, title=None, caption=None, rows=[]):
        """
        Parameters
        ----------
        columns : str
            alignment for each column, e.g. ``'lrr'``
        rules : bool
            Add toprule and bottomrule
        title : None | text
            Title for the table.
        caption : None | text
            Caption for the table.
        rows : list of Row
            Table body.
        """
        self.columns = columns
        self._table = rows[:]
        self.rules = rules
        self.title(title)
        self.caption(caption)
        self._active_row = None

    @property
    def shape(self):
        return (len(self._table), len(self.columns))

    def __len__(self):
        return len(self._table)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._table[item]
        else:
            rows = self._table[item]
            return Table(self.columns, rules=self.rules, title=self._title,
                         caption=self._caption, rows=rows)

    # adding texstrs ---
    def cell(self, *args, **kwargs):
        """
        args:   text, *properties
        OR:     FMText object

        properties are tex text properties (e.g. "textbf")


        kwargs
        ------

        Cell kwargs, e.g.:
        width=1     use value >1 for multicolumn cells
        just='l'    justification (only for multicolumn)
        mat=False   enclose tex output in $...$ if True
        FMText kwargs ...


        number properties
        -----------------

        drop0=False     drop 0 before
        digits=4        number of digits after dot


        Properties Example
        ------------------
        >>> table.cell("Entry", "textsf", "textbf") for bold sans serif
        """
        if len(args) == 0:
            txt = ''
        else:
            txt = args[0]

        if len(args) > 1:
            property = args[1]
        else:
            property = None

        cell = Cell(text=txt, property=property, **kwargs)

        if self._active_row is None or len(self._active_row) == len(self.columns):
            new_row = Row()
            self._table.append(new_row)
            self._active_row = new_row

        if len(cell) + len(self._active_row) > len(self.columns):
            raise ValueError("Cell too long -- row width exceeds table width")
        self._active_row.append(cell)

    def empty_row(self):
        self.endline()
        self._table.append(Row())

    def endline(self):
        "finishes the active row"
        if self._active_row is not None:
            for _ in xrange(len(self.columns) - len(self._active_row)):
                self._active_row.append(Cell())
        self._active_row = None

    def cells(self, *cells):
        "add several simple cells with one command"
        for cell in cells:
            self.cell(cell)

    def midrule(self, span=None):
        """
        adds midrule; span ('2-4' or (2, 4)) specifies the extent

        note that a toprule and a bottomrule are inserted automatically
        in every table.
        """
        self.endline()
        if span is None:
            self._table.append("\\midrule")
        else:
            if type(span) in [list, tuple]:
                # TODO: assert end is not too big
                span = '-'.join([str(int(i)) for i in span])
            assert '-' in span
            assert all([i.isdigit() for i in span.split('-')])
            self._table.append(r"\cmidrule{%s}" % span)

    def title(self, *args, **kwargs):
        """Set the table title (with FMText args/kwargs)"""
        if (len(args) == 1) and (args[0] is None):
            self._title = None
        else:
            self._title = FMText(*args, **kwargs)

    def caption(self, *args, **kwargs):
        """Set the table caption (with FMText args/kwargs)"""
        if (len(args) == 1) and (args[0] is None):
            self._caption = None
        else:
            self._caption = FMText(*args, **kwargs)

    def __repr__(self):
        """
        return self.__str__ so that when a function returns a Table, the
        result can be inspected without assigning the Table to a variable.

        """
        return self.__str__()

    def get_html(self, options={}):
        if self._caption is None:
            caption = None
        else:
            if preferences['html_tables_in_fig']:
                tag = 'figcaption'
            else:
                tag = 'caption'
            caption = _html_element(tag, self._caption)

        # table body
        table = []
        if caption and not preferences['html_tables_in_fig']:
            table.append(caption)
        for row in self._table:
            if isstr(row):
                if row == "\\midrule":
                    pass
#                     table.append('<tr style="border-bottom:1px solid black">')
            else:
                table.append(row.get_html(options))
        body = '\n'.join(table)

        # table frame
        if self.rules:
            table_options = {'border': 1, 'frame': 'hsides', 'rules': 'none'}
        else:
            table_options = {'border': 0}
        txt = _html_element('table', body, table_options)

        # embedd in a figure
        if preferences['html_tables_in_fig']:
            if caption:
                txt = '\n'.join((txt, caption))
            txt = _html_element('figure', txt)

        return txt

    def get_str(self, options={}, delim='   ', linesep=os.linesep):
        """Convert Table to str

        Parameters
        ----------
        fmt : None  | str
            Format for numbers.
        delim : str
            Delimiter between columns.
        linesep : str
            Line separation string
        """
        # append to recent tex out
        _add_to_recent(self)

        # determine column widths
        widths = []
        for row in self._table:
            if not isstr(row):  # some commands are str
                row_strlen = row._strlen(options)
                while len(row_strlen) < len(self.columns):
                    row_strlen.append(0)
                widths.append(row_strlen)
        try:
            widths = np.array(widths)
        except Exception, exc:
            print widths
            raise Exception(exc)
        c_width = np.max(widths, axis=0)  # column widths!

        # FIXME: take into account tab length:
        midrule = delim.join(['-' * w for w in c_width])
        midrule = midrule.expandtabs(4).replace(' ', '-')

        # collect lines
        txtlines = []
        for row in self._table:
            if isstr(row):  # commands
                if row == r'\midrule':
                    txtlines.append(midrule)  # "_"*l_len)
                elif row == r'\bottomrule':
                    txtlines.append(midrule)  # "_"*l_len)
                elif row == r'\toprule':
                    txtlines.append(midrule)  # "_"*l_len)
                elif row.startswith(r'\cmidrule'):
                    txt = row.split('{')[1]
                    txt = txt.split('}')[0]
                    start, end = txt.split('-')
                    start = int(start) - 1
                    end = int(end)
                    line = [' ' * w for w in c_width[:start]]
                    rule = delim.join(['-' * w for w in c_width[start:end]])
                    rule = rule.expandtabs(4).replace(' ', '-')
                    line += [rule]
                    line += [' ' * w for w in c_width[start:end]]
                    txtlines.append(delim.join(line))
                else:
                    pass
            else:
                txtlines.append(row.get_str(c_width, self.columns, delim,
                                            options))
        out = txtlines

        if self._title != None:
            out = ['', self._title.get_str(options), ''] + out

        if isstr(self._caption):
            out.append(self._caption)
        elif self._caption:
            out.append(str(self._caption))

        return linesep.join(out)

    def get_tex(self, options={}):
        tex_pre = [r"\begin{center}",
                   r"\begin{tabular}{%s}" % self.columns]
        if self.rules:
            tex_pre.append(r"\toprule")
        # Body
        tex_body = []
        for row in self._table:
            if isstr(row):
                tex_body.append(row)
            else:
                tex_body.append(row.get_tex(options))
        # post
        tex_post = [r"\end{tabular}",
                    r"\end{center}"]
        if self.rules:
            tex_post = [r"\bottomrule"] + tex_post
        # combine
        tex_repr = os.linesep.join(tex_pre + tex_body + tex_post)
        return tex_repr

    def get_tsv(self, delimiter='\t', linesep='\r\n', fmt='%.9g'):
        """
        Returns the table as tsv string.

        kwargs
        ------
        delimiter: delimiter between columns (by default tab)
        linesep:   delimiter string between lines
        fmt:       format for numerical entries
        """
        table = []
        for row in self._table:
            if isstr(row):
                pass
            else:
                table.append(row.get_tsv(delimiter, fmt=fmt))
        return linesep.join(table)

    def copy_pdf(self):
        "copy pdf to clipboard"
        copy_pdf(self)

    def copy_tex(self):
        "copy tex t clipboard"
        copy_tex(self)

    def save_pdf(self, path=None):
        "saves table on pdf; if path == non ask with system dialog"
        save_pdf(self, path=path)

    def save_tex(self, path=None):
        "saves table as tex; if path == non ask with system dialog"
        save_tex(self, path=path)

    def save_tsv(self, path=None, delimiter='\t', linesep='\r\n', fmt='%.15g'):
        """
        Save the table as tab-separated values file.

        Parameters
        ----------
        path : str | None
            Destination file name.
        delimiter : str
            String that is placed between cells (default: tab).
        linesep : str
            String that is placed in between lines.
        fmt : str
            Format string for representing numerical cells.
            (see 'Python String Formatting Documentation
            <http://docs.python.org/library/stdtypes.html#string-formatting-operations>'_ )
        """
        if not path:
            path = ui.ask_saveas(title="Save Tab Separated Table",
                                 message="Please Pick a File Name",
                                 ext=[("txt", "txt (tsv) file")])
        if ui.test_targetpath(path):
            ext = os.path.splitext(path)[1]
            if ext == '':
                path += '.txt'

            with open(path, 'w') as f:
                out = self.get_tsv(delimiter=delimiter, linesep=linesep,
                                   fmt=fmt)
                if isinstance(out, unicode):
                    out = out.encode('utf-8')
                f.write(out)

    def save_txt(self, path=None, fmt='%.15g', delim='   ', linesep=os.linesep):
        """
        Save the table as text file.

        Parameters
        ----------
        path : str | None
            Destination file name.
        fmt : str
            Format string for representing numerical cells.
        linesep : str
            String that is placed in between lines.
        """
        if not path:
            path = ui.ask_saveas(title="Save Table as Text File",
                                 message="Please Pick a File Name",
                                 ext=[("txt", "txt file")])
        if ui.test_targetpath(path):
            ext = os.path.splitext(path)[1]
            if ext == '':
                path += '.txt'

            with open(path, 'w') as f:
                options = {'fmt': fmt}
                out = self.get_str(options, delim, linesep)
                if isinstance(out, unicode):
                    out = out.encode('utf-8')
                f.write(out)


class Image(FMTextElement, StringIO):
    "Represent an image file"

    def __init__(self, filename, alt=None, buf=''):
        """Represent an image file

        Parameters
        ----------
        filename : str
            Filename for the image file (should have the appropriate
            extension). If a document has multiple images with the same name,
            a unique integer is appended.
        alt : None | str
            Alternate text, placeholder in case the image can not be found
            (HTML `alt` tag).
        """
        StringIO.__init__(self, buf)

        self._filename = filename
        self._alt = alt or filename

    @classmethod
    def from_file(self, path, filename=None, alt=None):
        """Create an Image object from an existsing image file.

        Parameters
        ----------
        path : str
            Path to the image file.
        filename : None | str
            Filename for the target image. The default is
            os.path.basename(path).
        alt : None | str
            Alternate text, placeholder in case the image can not be found
            (HTML `alt` tag).
        """
        if filename is None:
            filename = os.path.basename(path)

        with open(path, 'rb') as fid:
            buf = fid.read()

        return Image(filename, alt, buf)

    def __repr_items__(self):
        out = [repr(self._filename)]
        if self._alt != self._filename:
            out.append(repr(self._alt))
        v = self.getvalue()
        if len(v) > 0:
            out.append('buf=%s...' % repr(v[:50]))
        return out

    def get_html(self, options={}):
        dirpath = os.path.join(options['root'], options['resource_dir'])
        abspath = os.path.join(dirpath, self._filename)
        if os.path.exists(abspath):
            i = 0
            name, ext = os.path.splitext(self._filename)
            while os.path.exists(abspath):
                i += 1
                filename = name + ' %i' % i + ext
                abspath = os.path.join(dirpath, filename)

        self.save_image(abspath)

        relpath = os.path.relpath(abspath, options['root'])
        txt = ' <img src="%s" alt="%s">' % (relpath, html(self._alt))
        return ' ' + txt

    def get_str(self, options={}):
        txt = "Image (%s)" % str(self._alt)
        return txt

    def save_image(self, dst):
        if os.path.isdir(dst):
            dst = os.path.join(dst, self._filename)

        buf = self.getvalue()
        with open(dst, 'wb') as fid:
            fid.write(buf)


class Figure(FMText):
    "Represent a figure"

    def __init__(self, content, caption=None):
        """Represent a figure

        Parameters
        ----------
        content : Text
        """
        self._caption = caption
        FMText.__init__(self, content)

    def get_html(self, options={}):
        body = FMText.get_html(self, options)
        if self._caption:
            caption = _html_element('figcaption', self._caption)
            body = '\n'.join((body, caption))
        txt = _html_element('figure', body)
        return txt

    def get_str(self, options={}):
        body = FMText.get_str(self, options)
        caption = str(self._caption)
        txt = "\nFigure:\n%s\nCaption: %s" % (body, caption)
        return txt


class Section(FMText):

    def __init__(self, heading, content=[]):
        """Represent a section of an FMText document

        Parameters
        ----------
        heading : FMText
            Section heading.
        content : list of FMText
            Section content. Can also be constructed dynamically through the
            different .add_... methods.
        """
        self._heading = heading
        FMText.__init__(self, content)

    def add_image_figure(self, filename, caption, alt=None):
        """Add an image in a figure frame to the section

        Parameters
        ----------
        filename : str
            Filename for the image file (should have the appropriate
            extension). If a document has multiple images with the same name,
            a unique integer is appended.
        caption : FMText
            Image caption.
        alt : None | str
            Alternate text, placeholder in case the image can not be found
            (HTML `alt` tag).

        Returns
        -------
        image : Image
            Image object that was added.
        """
        image = Image(filename, alt)
        figure = Figure(image, caption)
        self.append(figure)
        return image

    def add_section(self, heading, content=[]):
        """Add a new subordinate section

        Parameters
        ----------
        heading : FMText
            Heading for the section.
        content : None | list of FMText
            Content for the section.

        Returns
        -------
        section : Section
            The new section.
        """
        section = Section(heading, content)
        self.append(section)
        return section

    def get_html(self, options={}):
        options = options.copy()
        options['level'] = options.get('level', 1)
        heading = self._get_html_section_heading(options)

        options['level'] += 1
        body = FMText.get_html(self, options)

        txt = '\n\n'.join(('', heading, body))
        return txt

    def _get_html_section_heading(self, options):
        level = options['level']
        tag = 'h%i' % level
        heading = '<%s>%s</%s>' % (tag, html(self._heading), tag)
        return heading

    def get_str(self, options={}):
        level = options.get('level', (1,))
        number = '.'.join(map(str, level))
        title = ' '.join((number, str(self._heading)))
        if len(level) == 1:
            underline_char = '='
        else:
            underline_char = '-'
        underline = underline_char * len(title)

        content = [title, underline, '']
        options = options.copy()
        level = list(level) + [1]
        for item in self._content:
            if isinstance(item, Section):
                options['level'] = tuple(level)
                txt = item.get_str(options)
                level[-1] += 1
                content += ['', '', txt]
            else:
                content += [str(item)]

        txt = '\n'.join(content)
        return txt

    def get_title(self):
        return self._heading


class Report(Section):

    def __init__(self, title, author=None, date=True, content=[]):
        """Represent an FMText report document

        Parameters
        ----------
        title : FMText
            Document title.
        author : None | FMText
            Document autho.
        date : None | True | FMText
            Date to print on the report. If True, the current day (object
            initialization) is used.
        content : list of FMText
            Report content. Can also be constructed dynamically through the
            different .add_... methods.
        """
        if author is not None:
            author = FMText(author, r'\author')
        if date is not None:
            if date is True:
                date = str(datetime.date.today())
            date = FMText(date, r'\date')
        self._author = author
        self._date = date
        Section.__init__(self, title, content)

    def _get_html_section_heading(self, options):
        level = options['level']
        if level != 1:
            raise ValueError("Report must be top level.")

        content = []
        if self._heading is not None:
            title = _html_element('h1', self._heading)
            content.append(title)
        if self._author is not None:
            author = html(self._author, options)
            content.append(author)
        if self._date is not None:
            date = html(self._date, options)
            content.append(date)

        txt = '\n\n'.join(content)
        return txt

    def get_str(self, options={}):
        content = []
        if self._heading is not None:
            title = str(self._heading)
            underline = '^' * len(title)
            content += [title, underline, '']
        if self._author is not None:
            author = self._author.get_str(options)
            content += [author, '']
        if self._date is not None:
            date = self._date.get_str(options)
            content += [date, '']

        if content:
            content += ['', '']

        level = [1]
        options = options.copy()
        for item in self._content:
            if isinstance(item, Section):
                options['level'] = tuple(level)
                txt = item.get_str(options)
                level[-1] += 1
            else:
                txt = item.get_str(options)
            content += [txt, '']

        txt = '\n'.join(content)
        return txt

    def pickle(self, path, extension='.pickled'):
        """Pickle the Report object

        Parameters
        ----------
        path : str
            Location where to save the report. For None, the file is saved
            in the report's folder.
        extension : None | str
            Extension to append to the path. If extension is None, or path
            already ends with extension nothing is done.
        """
        if extension and not path.endswith(extension):
            path += extension

        with open(path, 'wb') as fid:
            pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)

    def save_html(self, path, pickle=True):
        """Save HTML file of the report

        Parameters
        ----------
        path : str
            Path at which to save the html. Does not need to contain the
            extension. A folder with the same name is created fo resource
            files.
        pickle : bool
            Also store a pickled version of the report in the resource folder.
        """
        if path.endswith('.html'):
            path = path[:-5]

        save_html(self, path)

        if pickle:
            name = os.path.basename(path)
            pickle_path = os.path.join(path, name)
            self.pickle(pickle_path)


def unindent(text, skip1=False):
    """Removes leading spaces that are present in all lines of ``text``.

    Parameters
    ----------
    test : str
        The text from which leading spaces should be removed.
    skip1 : bool
        Ignore the first line when determining number of spaces to unindent,
        and remove all leading whitespaces from it.
    """
    # count leading whitespaces
    lines = text.splitlines()
    ws_lead = []
    for line in lines[skip1:]:
        len_stripped = len(line.lstrip(' '))
        if len_stripped:
            ws_lead.append(len(line) - len_stripped)

    if len(ws_lead) > skip1:
        rm = min(ws_lead)
        if rm:
            if skip1:
                lines[0] = ' ' * rm + lines[0].lstrip()

            text = os.linesep.join(line[rm:] for line in lines)

    return text
