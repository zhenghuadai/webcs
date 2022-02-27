
/*
function displayMatrix(matrix, name, el)
{
	let rows = matrix[0];
	let cols = matrix[1];
	let idx  = 2;
	__displayMatrix(matrix, name, el, rows, cols, idx)
}
*/

function displayMatrix(matrix, name, el, rows, cols, idxBase)
{
	/*
        <table class="w3-large">
        <tbody><tr>
        <td>C =&nbsp;&nbsp;</td>
        <td>
        <table class="matrix">
        <tbody><tr>
          <td>2</td> <td>5</td> <td>3</td>
        </tr>
        <tr>
          <td>4</td> <td>7</td> <td>1</td>
        </tr>
        </tbody></table>
        </td>
        </tr>
        </tbody></table>
        */
	if (idxBase == undefined)
	{
		idxBase = 0;
	}
	let str      = [];
	const MAXCOL = 6;
	const MAXROW = 6;
    function makestr(f) { return f.toFixed(8);}
	if ((rows < MAXROW) && (cols < MAXCOL))
	{
		let idx = idxBase;
		for (var row = 0; row < rows; row++)
		{
			str.push('<tr>');
			for (var col = 0; col < cols; col++)
			{
				str.push('<td>' + makestr(matrix[idx]) + ' </td>');
				idx = idx + 1;
			}
			str.push('</tr>');
		}
	}
	else
	{
		function addRow(row)
		{
			const idx = idxBase;
			str.push('<tr>');
			if (cols > MAXCOL)
			{
				str.push('<td>' + makestr(matrix[idx + row * cols + 0]) + ' </td>');
				str.push('<td>' + makestr(matrix[idx + row * cols + 1]) + ' </td>');
				str.push('<td>' + makestr(matrix[idx + row * cols + 2]) + ' </td>');
				str.push('<td> ... </td>');
				str.push('<td>' + makestr(matrix[idx + row * cols + cols - 3]) + ' </td>');
				str.push('<td>' + makestr(matrix[idx + row * cols + cols - 2]) + ' </td>');
				str.push('<td>' + makestr(matrix[idx + row * cols + cols - 1]) + ' </td>');
			}
			else
			{
				for (var ii = 0; ii < cols; ii++)
				{
					str.push('<td>' + makestr(matrix[idx + row * cols + ii]) + ' </td>');
				}
			}
			str.push('</tr>');
		}
		function addEmptyRow()
		{
			str.push('<tr>');
			for (var ii = 0; ii < (cols < 7 ? cols : 7); ii++)
			{
				str.push('<td> ... </td>');
			}
			str.push('</tr>');
		}

		if (rows > 6)
		{
			addRow(0);
			addRow(1);
			addRow(2);
			addEmptyRow();
			addRow(rows - 3);
			addRow(rows - 2);
			addRow(rows - 1);
		}
		else
		{
			for (var jj = 0; jj < rows; jj++)
			{
				addRow(jj);
			}
		}
	}
	let matrixBody = '<table class=\'matrix\'><tbody>' + str.join('') + '</tbody></table>';

	let matrixHtml                        = `
            <table class="w3-large"><tbody><tr>
            <td>${name}(${rows}x${cols})  =&nbsp;&nbsp;</td>
            <td> ${matrixBody}</td>
            </tr></tbody></table>

            `;
    el.innerHTML = matrixHtml;
}

export {displayMatrix};
