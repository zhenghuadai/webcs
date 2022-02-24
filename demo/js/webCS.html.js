//(function() {
$('#image000').on('load', function() {
    refresh();
});
$(function() {
    let img = $('#image000');
    if (!img[0].complete) return;
    if ($('#image000').width() > 512) {
        $('#display0').children().addClass('cssize');
        refresh();
    } else if ($('#image000').width() > 0) {
        refresh();
    }
});

function refresh() {
    const canvas = $('#canvasimg0')[0];
    const img = $('#image000');
    let w = img.width();
    let h = img.height();
    canvas.width = img.width();
    canvas.height = img.height();
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img[0], 0, 0, w, h);
}


// file menu
(function() {
$('#loadimg').click(function() {
    $('#loadlocal').click();
});
$('#loadlocal').on('change', function(ev) {
    let input = ev.target;
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function(e) {
            $('#image000').attr('src', e.target.result);
        };

        reader.readAsDataURL(input.files[0]);
    }
});
$('#filemenu').menu();
})();


// view menu
$('#viewmenu').menu({
    select: function(e, ui) {
        let cmd = ui.item.attr('cmd');
        if (cmd == 0) {
        } else if (cmd == 1) {
        }
        if (cmd != undefined) {
        }
    }
});


// right buttons
$(function() {
    $('#AddonToggleButton').button();
    $('#GPUToggleButton').button();
    $('#timediv').button();
    $('#TestGPU').button();
});

// message box
function showMessage(txt) {
    $('#messagebox').html(txt);
    $('#messagebox').show();
    if (txt && txt.length > 0) {
        $('#messagebox').removeClass('minimize');
        $('#messagebox').addClass('ui-tooltip');
    } else {
        $('#messagebox').removeClass('ui-tooltip');
    }
    setTimeout(function() {
        $('#messagebox').addClass('minimize');
    }, 5000);
};
$('#messagebox')
    .hover(
        function() {
            $('#messagebox').removeClass('minimize');
        },
        function() {
            setTimeout(function() {
                $('#messagebox').addClass('minimize');
            }, 5000);
        });

$('#meesagebox').button();
let App = {};
App.showMessage = showMessage;
export {App};
//})();
