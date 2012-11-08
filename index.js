/* wrap everything in an anonymous function to prevent namespace pollution */
jQuery(function($){
var filters = {
    Elitism: {
        1: 'Yes',
        0: 'No'
    },
    Population: [
        10,
        20,
        50,
        100
    ],
    Mutation: [
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6
    ],
    Crossover: [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9
    ]
};

/* List of checkboxes checked by default, true means all, a (list of) values
 * means only those values */
var checked = {
    Elitism: 1,
    Population: 100,
    Mutation: [0.1, 0.2],
    Crossover: true
};

$.each(filters, function(filter, values){
    fieldset = $('<fieldset class="inline">');
    fieldset.append('<legend class="inline">' + filter + '</legend>');
    var name = filter.toLowerCase();

    $.each(values, function(value, label){
        if(!isNaN(label)){
            value = label;
        }
        fieldset.append(
            '<label class="checkbox inline">'
            + '<input type="checkbox" name="' + name + '" value="' + value + '">'
            + label
        );
    });
    var checkboxes = ':checkbox';
    var check = checked[filter];
    if(!check){
        // no checkboxes need to be checked so default to the last item
        checkboxes += ' :last';
    }else if(check === true){
        // all checkboxes, no need for filters
    }else if($.isArray(check)){
        // all values in the given list
        var filters = [];
        $.each(check, function(i, c){
            filters.push(checkboxes + '[value="' + c + '"]');
        });
        checkboxes = filters.join(',');
    }else{
        checkboxes += '[value="' + check + '"]';
    }

    fieldset.find(checkboxes).attr('checked', 'checked');

    $('form#filterform').append(fieldset);
});

var images = $('#images');
for(var elitism in filters.Elitism){
    for(var p_i in filters.Population){
        var population = filters.Population[p_i];
        for(var m_i in filters.Mutation){
            var mutation = filters.Mutation[m_i];
            for(var c_i in filters.Crossover){
                var crossover  = filters.Crossover[c_i];
                var src = 'images/'
                    + 'elitism_' + elitism
                    +'_population_' + population
                    +'_mutation_' + mutation
                    +'_crossover_' + crossover
                    + '.png';
                var image = $('<img>').attr({
                    elitism: elitism,
                    population: population,
                    mutation: mutation,
                    crossover: crossover,
                    src: src,
                    style: ''
                        + 'float: left;'
                        + 'display: none;',
                    title: ''
                        + 'Elitism: ' + filters.Elitism[elitism] + '<br>'
                        + 'Population: ' + population + '<br>'
                        + 'Mutation: ' + mutation + '<br>'
                        + 'Crossover: ' + crossover
                });
                images.append(image);
            }
        }
    }
}

$.fn.serializeObject = function()
{
    var o = {};
    var a = this.serializeArray();
    $.each(a, function() {
        if (o[this.name] !== undefined) {
            if (!o[this.name].push) {
                o[this.name] = [o[this.name]];
            }
            o[this.name].push(this.value || '');
        } else {
            o[this.name] = this.value || '';
        }
    });
    return o;
};

function change(){
    $('img').hide();
    var filters = [];
    $('input[name=elitism]:checked').each(function(_, elitism){
        $('input[name=population]:checked').each(function(_, population){
            $('input[name=mutation]:checked').each(function(_, mutation){
                $('input[name=crossover]:checked').each(function(_, crossover){
                    filters.push('img'
                        + '[elitism="' + elitism.value +'"]'
                        + '[population="' + population.value +'"]'
                        + '[mutation="' + mutation.value +'"]'
                        + '[crossover="' + crossover.value +'"]'
                    );
                });
            });
        });
    });
    $(filters.join(',')).show();
}

$('input').change(change);
change();
$('img').tooltip({
    html: true,
    placement: 'bottom'
});

});
