/*=========================================================================================
  File Name: app.js
  Description: Template related app JS.
  ----------------------------------------------------------------------------------------
  Item Name: Chameleon Admin - Modern Bootstrap 4 WebApp & Dashboard HTML Template + UI Kit
  Version: 1.0
  Author: ThemeSelection
  Author URL: https://themeforest.net/user/themeselect
==========================================================================================*/

(function(window, document, $) {
    'use strict';
    var $html = $('html');
    var $body = $('body');


    $(window).on('load', function() {
        var rtl;
        var compactMenu = false; // Set it to true, if you want default menu to be compact

        if ($('html').data('textdirection') == 'rtl') {
            rtl = true;
        }

        setTimeout(function() {
            $html.removeClass('loading').addClass('loaded');
        }, 1200);

        $.app.menu.init(compactMenu);

        // Navigation configurations
        var config = {
            speed: 300 // set speed to expand / collpase menu
        };
        if ($.app.nav.initialized === false) {
            $.app.nav.init(config);
        }

        Unison.on('change', function(bp) {
            $.app.menu.change();
        });

        var $sidebar_img = $('.main-menu').data('img'),
        $sidebar_img_container = $('.navigation-background');

        if( $sidebar_img_container.length > 0 && $sidebar_img !== undefined ){
            $sidebar_img_container.css('background-image','url("' + $sidebar_img + '")');
        }

        // Tooltip Initialization
        $('[data-toggle="tooltip"]').tooltip({
            container: 'body'
        });

        //Hide navbar search box on close click
        var toogleBtn = $(".header-navbar .navbar-search-close");
        $(toogleBtn).click(function(event) {
            $('.navbar-search .dropdown-toggle').click();
        });

        // Top Navbars - Hide on Scroll
        if ($(".navbar-hide-on-scroll").length > 0) {
            $(".navbar-hide-on-scroll.fixed-top").headroom({
                "offset": 205,
                "tolerance": 5,
                "classes": {
                    // when element is initialised
                    initial: "headroom",
                    // when scrolling up
                    pinned: "headroom--pinned-top",
                    // when scrolling down
                    unpinned: "headroom--unpinned-top",
                }
            });
            // Bottom Navbars - Hide on Scroll
            $(".navbar-hide-on-scroll.fixed-bottom").headroom({
                "offset": 205,
                "tolerance": 5,
                "classes": {
                    // when element is initialised
                    initial: "headroom",
                    // when scrolling up
                    pinned: "headroom--pinned-bottom",
                    // when scrolling down
                    unpinned: "headroom--unpinned-bottom",
                }
            });
        }

        //Match content & menu height for content menu
        setTimeout(function() {
            if ($('body').hasClass('vertical-content-menu')) {
                setContentMenuHeight();
            }
        }, 500);

        function setContentMenuHeight() {
            var menuHeight = $('.main-menu').height();
            var bodyHeight = $('.content-body').height();
            if (bodyHeight < menuHeight) {
                $('.content-body').css('height', menuHeight);
            }
        }

        // Collapsible Card
        $('a[data-action="collapse"]').on('click', function(e) {
            e.preventDefault();
            $(this).closest('.card').children('.card-content').collapse('toggle');
            $(this).closest('.card').find('[data-action="collapse"] i').toggleClass('ft-minus ft-plus');

        });

        // Toggle fullscreen
        $('a[data-action="expand"]').on('click', function(e) {
            e.preventDefault();
            $(this).closest('.card').find('[data-action="expand"] i').toggleClass('ft-maximize ft-minimize');
            $(this).closest('.card').toggleClass('card-fullscreen');
        });

        //  Notifications & messages scrollable
        if ($('.scrollable-container').length > 0) {
            $('.scrollable-container').perfectScrollbar({
                theme: "dark"
            });
        }

        // Reload Card
        $('a[data-action="reload"]').on('click', function() {
            var block_ele = $(this).closest('.card');

            // Block Element
            block_ele.block({
                message: '<div class="ft-refresh-cw icon-spin font-medium-2"></div>',
                timeout: 2000, //unblock after 2 seconds
                overlayCSS: {
                    backgroundColor: '#FFF',
                    cursor: 'wait',
                },
                css: {
                    border: 0,
                    padding: 0,
                    backgroundColor: 'none'
                }
            });
        });

        // Close Card
        $('a[data-action="close"]').on('click', function() {
            $(this).closest('.card').removeClass().slideUp('fast');
        });

        // Match the height of each card in a row
        setTimeout(function() {
            $('.row.match-height').each(function() {
                $(this).find('.card').not('.card .card').matchHeight(); // Not .card .card prevents collapsible cards from taking height
            });
        }, 500);


        $('.card .heading-elements a[data-action="collapse"]').on('click', function() {
            var $this = $(this),
                card = $this.closest('.card');
            var cardHeight;

            if (parseInt(card[0].style.height, 10) > 0) {
                cardHeight = card.css('height');
                card.css('height', '').attr('data-height', cardHeight);
            } else {
                if (card.data('height')) {
                    cardHeight = card.data('height');
                    card.css('height', cardHeight).attr('data-height', '');
                }
            }
        });

        // Add open class to parent list item if subitem is active except compact menu
        var menuType = $body.data('menu');
        if (menuType != 'vertical-compact-menu' && menuType != 'horizontal-menu' && compactMenu === false) {
            if ($body.data('menu') == 'vertical-menu-modern') {
                if (localStorage.getItem("menuLocked") === "true") {
                    $(".main-menu-content").find('li.active').parents('li').addClass('open');
                }
            } else {
                $(".main-menu-content").find('li.active').parents('li').addClass('open');
            }
        }
        if (menuType == 'vertical-compact-menu' || menuType == 'horizontal-menu') {
            $(".main-menu-content").find('li.active').parents('li:not(.nav-item)').addClass('open');
            $(".main-menu-content").find('li.active').parents('li').addClass('active');
        }

        //card heading actions buttons small screen support
        $(".heading-elements-toggle").on("click", function() {
            $(this).parent().children(".heading-elements").toggleClass("visible");
        });

        //  Dynamic height for the chartjs div for the chart animations to work
        var chartjsDiv = $('.chartjs'),
            canvasHeight = chartjsDiv.children('canvas').attr('height');
        chartjsDiv.css('height', canvasHeight);

        if ($body.hasClass('boxed-layout')) {
            if ($body.hasClass('vertical-overlay-menu') || $body.hasClass('vertical-compact-menu')) {
                var menuWidth = $('.main-menu').width();
                var contentPosition = $('.app-content').position().left;
                var menuPositionAdjust = contentPosition - menuWidth;
                if ($body.hasClass('menu-flipped')) {
                    $('.main-menu').css('right', menuPositionAdjust + 'px');
                } else {
                    $('.main-menu').css('left', menuPositionAdjust + 'px');
                }
            }
        }

       
    });


    $(document).on('click', '.menu-toggle, .modern-nav-toggle', function(e) {
        e.preventDefault();

        // Toggle menu
        $.app.menu.toggle();

        setTimeout(function() {
            $(window).trigger("resize");
        }, 200);

        if ($('#collapsed-sidebar').length > 0) {
            setTimeout(function() {
                if ($body.hasClass('menu-expanded') || $body.hasClass('menu-open')) {
                    $('#collapsed-sidebar').prop('checked', false);
                } else {
                    $('#collapsed-sidebar').prop('checked', true);
                }
            }, 1000);
        }

        return false;
    });

    $(document).on('click', '.close-navbar', function(e) {
        e.preventDefault();

        // Toggle menu
        $.app.menu.toggle();
    });

    $(document).on('click', '.open-navbar-container', function(e) {

        var currentBreakpoint = Unison.fetch.now();

        // Init drilldown on small screen
        $.app.menu.drillDownMenu(currentBreakpoint.name);

        // return false;
    });


    // Add Children Class
    $('.navigation').find('li').has('ul').addClass('has-sub');

    $('.carousel').carousel({
        interval: 2000
    });


    $(document).on('click', '.mega-dropdown-menu', function(e) {
        e.stopPropagation();
    });

    // Update manual scroller when window is resized
    $(window).resize(function() {
        $.app.menu.manualScroller.updateHeight();
    });

})(window, document, jQuery);

//全选
$(".allsel").on('click', function() {
    $("tbody input:checkbox").prop("checked", $(this).prop('checked'));
})
$("tbody input:checkbox").on('click', function() {
    //当选中的长度等于checkbox的长度的时候,就让控制全选反选的checkbox设置为选中,否则就为未选中
    if ($("tbody input:checkbox").length === $("tbody input:checked").length) {
        $(".allsel").prop("checked", true);
    } else {
        $(".allsel").prop("checked", false);
    }
})